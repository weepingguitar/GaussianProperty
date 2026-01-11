from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


_PLY_TO_NUMPY = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "float32": np.float32,
    "float64": np.float64,
}

_NUMPY_TO_PLY = {
    np.dtype(np.int8): "char",
    np.dtype(np.uint8): "uchar",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "ushort",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}


@dataclass
class PlyVertexData:
    header_lines: List[str]
    vertex_properties: List[Tuple[str, str]]  # (ply_type, name)
    vertex: np.ndarray  # structured array
    format: str  # ascii | binary_little_endian | binary_big_endian


def _parse_header(fp) -> Tuple[List[str], str, int, List[Tuple[str, str]], int]:
    header_lines: List[str] = []
    fmt = None
    vertex_count = None
    vertex_props: List[Tuple[str, str]] = []

    in_vertex_element = False

    while True:
        line = fp.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header")
        try:
            s = line.decode("ascii").rstrip("\n")
        except UnicodeDecodeError:
            s = line.decode("utf-8", errors="ignore").rstrip("\n")

        header_lines.append(s)

        if s.startswith("format "):
            fmt = s.split()[1]
        elif s.startswith("element vertex "):
            vertex_count = int(s.split()[2])
            in_vertex_element = True
        elif s.startswith("element "):
            if not s.startswith("element vertex "):
                in_vertex_element = False
        elif in_vertex_element and s.startswith("property "):
            parts = s.split()
            if parts[1] == "list":
                raise NotImplementedError("PLY list properties not supported")
            ply_type = parts[1]
            name = parts[2]
            vertex_props.append((ply_type, name))
        elif s.strip() == "end_header":
            break

    if fmt is None or vertex_count is None:
        raise ValueError("Invalid PLY header: missing format or vertex count")

    data_start = fp.tell()
    return header_lines, fmt, vertex_count, vertex_props, data_start


def read_ply_vertices(path: str | Path) -> PlyVertexData:
    path = Path(path)
    with path.open("rb") as fp:
        header_lines, fmt, vertex_count, vertex_props, data_start = _parse_header(fp)

        if fmt == "ascii":
            # Avoid implementing ASCII parsing unless needed.
            raise NotImplementedError("ASCII PLY not supported; please export binary_little_endian")
        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise NotImplementedError(f"Unsupported PLY format: {fmt}")

        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype_fields = []
        for ply_type, name in vertex_props:
            if ply_type not in _PLY_TO_NUMPY:
                raise NotImplementedError(f"Unsupported PLY scalar type: {ply_type}")
            dtype_fields.append((name, endian + np.dtype(_PLY_TO_NUMPY[ply_type]).str[1:]))
        dtype = np.dtype(dtype_fields)

        fp.seek(data_start)
        vertex = np.fromfile(fp, dtype=dtype, count=vertex_count)

    return PlyVertexData(
        header_lines=header_lines,
        vertex_properties=vertex_props,
        vertex=vertex,
        format=fmt,
    )


def append_vertex_fields(vertex: np.ndarray, new_fields: Dict[str, np.ndarray], dtypes: Dict[str, np.dtype]) -> np.ndarray:
    if not isinstance(vertex.dtype, np.dtype) or vertex.dtype.fields is None:
        raise ValueError("vertex must be a structured numpy array")

    n = vertex.shape[0]
    for name, arr in new_fields.items():
        if arr.shape[0] != n:
            raise ValueError(f"Field {name} has length {arr.shape[0]} but vertex has {n}")

    old_descr = list(vertex.dtype.descr)
    for name in new_fields.keys():
        if name in vertex.dtype.names:
            raise ValueError(f"Field already exists in PLY: {name}")
        dt = np.dtype(dtypes[name])
        old_descr.append((name, dt.str))

    new_vertex = np.empty(n, dtype=np.dtype(old_descr))
    for name in vertex.dtype.names:
        new_vertex[name] = vertex[name]
    for name, arr in new_fields.items():
        new_vertex[name] = arr

    return new_vertex


def write_ply_vertices(path: str | Path, data: PlyVertexData, vertex: np.ndarray, extra_vertex_props: List[Tuple[str, str]] | None = None) -> None:
    path = Path(path)

    if data.format not in ("binary_little_endian", "binary_big_endian"):
        raise NotImplementedError(f"Unsupported PLY format for writing: {data.format}")

    endian = "<" if data.format == "binary_little_endian" else ">"

    # Rebuild header, only for vertex properties.
    header_out: List[str] = []
    header_out.append("ply")
    header_out.append(f"format {data.format} 1.0")
    header_out.append(f"element vertex {vertex.shape[0]}")

    # Determine properties list based on vertex dtype order.
    props: List[Tuple[str, str]] = []
    for name in vertex.dtype.names:
        dt = np.dtype(vertex.dtype.fields[name][0])
        ply_t = _NUMPY_TO_PLY.get(dt)
        if ply_t is None:
            raise NotImplementedError(f"Cannot map numpy dtype to PLY type for field {name}: {dt}")
        props.append((ply_t, name))

    for ply_t, name in props:
        header_out.append(f"property {ply_t} {name}")

    header_out.append("end_header")

    with path.open("wb") as fp:
        fp.write(("\n".join(header_out) + "\n").encode("ascii"))
        # Ensure correct endianness
        v = vertex
        if (endian == "<" and v.dtype.byteorder == ">") or (endian == ">" and v.dtype.byteorder == "<"):
            v = v.byteswap().newbyteorder()
        v.tofile(fp)

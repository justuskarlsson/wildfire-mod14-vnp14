from collections import defaultdict
from datetime import date, datetime
import json
import math
import typing
from typing import TypeVar, TypedDict, Union, Type
import sqlite3
from sqlite3 import Connection

from torch import Tensor
from tqdm import tqdm

db: Connection = sqlite3.connect(":memory:")


def get_db():
    return db


T = TypeVar("T")
TYPE_MAPPING = {
    str: "TEXT",
    int: "INTEGER",
    float: "REAL",
    datetime: "TEXT",
    date: "TEXT",
}

FROM_DB = {
    str: str,
    int: int,
    float: float,
    datetime: lambda s: datetime.fromisoformat(s),
    date: lambda s: date.fromisoformat(s),
}


def to_db_datetime(dt: str | datetime):
    if isinstance(dt, str):
        return dt
    else:
        return dt.isoformat(sep=" ")


def to_db_date(dt: str | date):
    if isinstance(dt, str):
        return dt
    else:
        return dt.isoformat()


TO_DB = {
    str: str,
    int: int,
    float: float,
    datetime: to_db_datetime,
    date: to_db_date,
}


def deserialize_rows(Class: Type[T], rows, query: T):
    name_to_type = {}
    for name, field_type in Class.__annotations__.items():
        if name not in query:
            continue
        field_type = _resolve_field_type(field_type)
        name_to_type[name] = field_type
    field_types = []
    for key in query:
        field_types.append(name_to_type[key])
    deserialized_rows = []
    for row in rows:
        obj = {}
        for val, field_type, name in zip(row, field_types, query.keys()):
            try:
                if field_type in FROM_DB:
                    val = FROM_DB[field_type](val)
                else:
                    val = json.loads(val, cls=json.JSONDecoder)
                obj[name] = val
            except Exception as e:
                obj[name] = None
        deserialized_rows.append(obj)
    return deserialized_rows


def select(Class: Type[T], query: T):
    names = ", ".join(query.keys())
    table = Class.__name__
    statement = f"SELECT {names} FROM {table};"
    print(statement)
    rows = db.execute(statement).fetchall()
    # What if single?
    return deserialize_rows(Class, rows, query)


def select_where(Class: Type[T], query: T, where: str):
    names = ", ".join(query.keys())
    table = Class.__name__
    # ? Where might have to use param?
    statement = f"SELECT {names} FROM {table} WHERE {where};"
    print(statement)
    rows = db.execute(statement).fetchall()
    # What if single?
    return deserialize_rows(Class, rows, query)


def _serialize_field(val):
    t = _resolve_field_type(type(val))
    if t in TO_DB:
        val = TO_DB[t](val)
    elif isinstance(val, Tensor):
        val = val.item()
    else:
        val = json.dumps(val, cls=json.JSONEncoder)
    if isinstance(val, float) and math.isnan(val):
        val = 0.0
    return val


def _serialize_row(Class: Type[T], d: T):
    return [
        _serialize_field(val)
        for name, val in d.items()
        if name in Class.__annotations__
    ]


def insert(Class: Type[T], data: T | list[T]):
    is_many = isinstance(data, list)
    keys = data[0].keys() if is_many else data.keys()
    keys = [k for k in keys if k in Class.__annotations__]
    names = ", ".join(keys)
    placeholder = ", ".join(["?"] * len(keys))
    table = Class.__name__
    statement = f"INSERT INTO {table} ({names}) VALUES ({placeholder});"
    # print("insert", statement)
    if is_many:
        values = [_serialize_row(Class, d) for d in data]
        print(len(values), len(values[0]), len(keys))
        res = db.executemany(statement, values)
    else:
        values = _serialize_row(Class, data)
        res = db.execute(statement, values)
    db.commit()
    return res


def insert_many(Class: Type[T], data: list[T]):
    data_by_keys = defaultdict(list)
    for obj in data:
        data_by_keys[frozenset(obj.keys())].append(obj)
    for objs in data_by_keys.values():
        insert(Class, objs)


def _resolve_field_type(field_type):
    while args := typing.get_args(field_type):
        field_type = args[0]
        break
    return field_type


def is_optional(field_type):
    is_union = typing.get_origin(field_type) is Union
    return is_union and (type(None) in typing.get_args(field_type))


def create_table(Class: Type[T]) -> str:
    fields = []
    table_name = Class.__name__
    for field, field_type in Class.__annotations__.items():
        resolved_type = _resolve_field_type(field_type)
        sql_type = TYPE_MAPPING.get(resolved_type, "TEXT")
        null_constraint = "" if is_optional(field_type) else "NOT NULL"
        fields.append(f"{field} {sql_type} {null_constraint}".strip())
    fields_def = ", ".join(fields)
    stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({fields_def});"
    # print(stmt)
    db.execute(stmt)
    db.commit()


def drop_table(Class: Type[T]):
    table_name = Class.__name__
    stmt = f"DROP TABLE IF EXISTS {table_name};"
    db.execute(stmt)
    db.commit()


class InfoSample(TypedDict):
    run_id: int
    idx: int
    epoch: int
    dataset: str
    cur_num_fire: int
    next_num_fire: int
    tn: int
    fn: int
    fp: int
    tp: int
    iou: float
    f1: float
    precision: float
    recall: float


class Run(TypedDict):
    name: str


def init_db(path: str):
    global db
    db = sqlite3.connect(path)
    create_table(Run)
    create_table(InfoSample)

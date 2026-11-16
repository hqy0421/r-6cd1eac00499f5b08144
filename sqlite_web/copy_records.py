"""Copy selected rows (and the records they reference) between databases.

The source database is only read.  Rows are written to the target inside a
single transaction, so a constraint error leaves the target untouched.

Only ordinary tables with a single rowid-alias primary key (declared
``INTEGER PRIMARY KEY``) and single-column foreign keys are supported.  The
foreign-key graph formed by the tables involved in a copy must be acyclic and
those tables must not have triggers.  Those restrictions keep the
source -> target key mapping unambiguous: every copied row receives a fresh
primary key and every non-null foreign-key value is rewritten to the new key
of the referenced row.  NULL foreign keys stay NULL.

Structure compatibility is checked before anything is written: the target
must have every source column (same affinity) and exactly the same foreign
keys, and every referenced source record must exist.  A source row that
appears more than once in the selection is copied once; two distinct source
rows with identical contents are still copied separately.

The public entry point is :func:`copy_referenced_records`, which returns a
:class:`CopyResult` with per-table counts and the source -> target key
mapping.  :func:`preflight_copy` performs only the read-only validation.
"""

import json
import os
from dataclasses import dataclass, field


class CopyError(Exception):
    """A copy could not be performed. Nothing was written."""


# SQLite's column-affinity rules, used to compare source and target columns.
def _affinity(declared_type):
    value = (declared_type or '').upper()
    if 'INT' in value:
        return 'INTEGER'
    if 'CHAR' in value or 'CLOB' in value or 'TEXT' in value:
        return 'TEXT'
    if 'BLOB' in value or not value:
        return 'BLOB'
    if 'REAL' in value or 'FLOA' in value or 'DOUB' in value:
        return 'REAL'
    return 'NUMERIC'


def _lower(name):
    return name.lower()


def _quote(name):
    return '"%s"' % name.replace('"', '""')


def _dataset_label(dataset):
    for attr in ('filename', 'basename'):
        value = getattr(dataset, attr, None)
        if value:
            return value
    return _database_path(dataset) or repr(dataset)


def _database_path(dataset):
    """Best-effort on-disk identity for a DataSet-like object."""
    value = getattr(dataset, 'filename', None)
    if value:
        return value
    database = getattr(dataset, '_database', None)
    value = getattr(database, 'database', None) if database is not None else None
    if value:
        if value.startswith('file:'):
            value = value[5:].split('?', 1)[0]
        return value
    return None


def _check_integer_key(value, table, column):
    # Primary keys of supported tables are SQLite integers.  bool is an int
    # subclass but is never a valid row id, and floats/strs would make the
    # mapping silently lossy.
    if isinstance(value, bool) or not isinstance(value, int):
        raise CopyError('invalid integer key %r for %s.%s' %
                        (value, table, column))
    return value


@dataclass
class Column:
    name: str
    declared_type: str
    nullable: bool
    default: object
    primary_key: bool
    hidden: int

    @property
    def affinity(self):
        return _affinity(self.declared_type)

    @property
    def has_default(self):
        # SQLite reports "DEFAULT NULL" as the string 'NULL'.
        return self.default is not None and \
            str(self.default).strip().upper() != 'NULL'


@dataclass
class ForeignKey:
    column: str
    dest_table: str
    dest_column: str


@dataclass
class TableSchema:
    name: str
    columns: list
    primary_key: Column
    foreign_keys: list
    create_sql: str

    @property
    def column_map(self):
        return {_lower(column.name): column for column in self.columns}

    @property
    def non_pk_columns(self):
        return [column for column in self.columns if not column.primary_key]

    @property
    def fk_map(self):
        return {_lower(fk.column): fk for fk in self.foreign_keys}

    def column(self, name):
        return self.column_map.get(_lower(name))


@dataclass
class TableCopy:
    table: str
    count: int = 0
    mappings: list = field(default_factory=list)

    @property
    def source_keys(self):
        return [source for source, _ in self.mappings]

    @property
    def target_keys(self):
        return [target for _, target in self.mappings]

    def to_dict(self):
        return {
            'table': self.table,
            'count': self.count,
            'mappings': [{'source_pk': source, 'target_pk': target}
                         for source, target in self.mappings],
            'source_keys': self.source_keys,
            'target_keys': self.target_keys,
        }


@dataclass
class CopyResult:
    source: str
    target: str
    tables: list = field(default_factory=list)

    @property
    def total(self):
        return sum(table.count for table in self.tables)

    @property
    def counts(self):
        return {table.table: table.count for table in self.tables}

    @property
    def key_mapping(self):
        return {table.table: dict(table.mappings) for table in self.tables}

    # Descriptive aliases for callers that prefer these names.
    @property
    def source_to_target(self):
        return self.key_mapping

    @property
    def table_counts(self):
        return self.counts

    @property
    def source_name(self):
        return os.path.basename(self.source) if self.source else ''

    @property
    def target_name(self):
        return os.path.basename(self.target) if self.target else ''

    def mapping_items(self):
        for table in self.tables:
            for source, target in table.mappings:
                yield table.table, source, target

    def to_dict(self):
        return {
            'source': self.source,
            'target': self.target,
            'total': self.total,
            'counts': self.counts,
            'tables': [table.to_dict() for table in self.tables],
            # JSON object keys are strings, so the mapping is stringified.
            'mapping': {table.table: {str(source): target
                                      for source, target in table.mappings}
                        for table in self.tables},
        }


def _query(dataset, sql, params=()):
    return dataset.query(sql, params)


def _fetchone(dataset, sql, params=()):
    cursor = _query(dataset, sql, params)
    try:
        return cursor.fetchone()
    finally:
        cursor.close()


def _fetchall(dataset, sql, params=()):
    cursor = _query(dataset, sql, params)
    try:
        return cursor.fetchall()
    finally:
        cursor.close()


def _actual_table(dataset, table):
    """Resolve a table name case-insensitively and reject views/virtuals."""
    row = _fetchone(
        dataset,
        "SELECT name, type, COALESCE(sql, '') FROM sqlite_master "
        "WHERE lower(name) = lower(?) AND type IN ('table', 'view') "
        "ORDER BY type LIMIT 1",
        (table,))
    if row is None:
        raise CopyError('table "%s" does not exist in %s' %
                        (table, _dataset_label(dataset)))
    name, kind, sql = row
    if kind != 'table':
        raise CopyError('"%s" is a view, not a copyable table' % name)
    if sql.upper().lstrip().startswith('CREATE VIRTUAL TABLE'):
        raise CopyError('"%s" is a virtual table and is not supported' % name)
    return name


def load_table_schema(dataset, table):
    """Read and validate the supported structure of one ordinary table."""
    table = _actual_table(dataset, table)
    row = _fetchone(
        dataset,
        "SELECT COALESCE(sql, '') FROM sqlite_master "
        "WHERE type = 'table' AND name = ?", (table,))
    create_sql = row[0] if row else ''

    columns = []
    for col in _fetchall(dataset, 'PRAGMA table_xinfo(%s)' % _quote(table)):
        # (cid, name, type, notnull, dflt_value, pk, hidden)
        columns.append(Column(
            name=col[1],
            declared_type=col[2] or '',
            nullable=not bool(col[3]),
            default=col[4],
            primary_key=bool(col[5]),
            hidden=int(col[6] or 0)))

    pk_columns = [column for column in columns if column.primary_key]
    if len(pk_columns) != 1:
        raise CopyError(
            'table "%s" must have exactly one primary-key column' % table)
    primary_key = pk_columns[0]
    if primary_key.declared_type.upper() != 'INTEGER':
        raise CopyError(
            'table "%s" primary key must be declared INTEGER' % table)
    if 'WITHOUT ROWID' in create_sql.upper():
        raise CopyError(
            'WITHOUT ROWID tables are not supported: "%s"' % table)
    if any(column.hidden for column in columns):
        raise CopyError(
            'generated or hidden columns are not supported: "%s"' % table)
    # A plain INTEGER PRIMARY KEY aliases rowid and has no separate
    # primary-key index.  INTEGER PRIMARY KEY DESC (and similar forms) do
    # have one, and would need the key supplied on insert.
    for index in _fetchall(dataset, 'PRAGMA index_list(%s)' % _quote(table)):
        if len(index) > 3 and index[3] == 'pk':
            raise CopyError(
                'table "%s" primary key is not a rowid alias' % table)

    # PRAGMA foreign_key_list returns
    # (id, seq, table, from, to, on_update, on_delete, match).  Composite
    # keys share an id and have seq > 0.
    groups = {}
    for row in _fetchall(dataset,
                         'PRAGMA foreign_key_list(%s)' % _quote(table)):
        groups.setdefault(row[0], []).append(row)
    column_map = {_lower(column.name): column for column in columns}
    foreign_keys = []
    for rows in groups.values():
        if len(rows) != 1:
            raise CopyError(
                'composite foreign keys are not supported: "%s"' % table)
        row = rows[0]
        child_column = row[3]
        child = column_map.get(_lower(child_column))
        if child is None:
            raise CopyError('foreign key on unknown column %s.%s' %
                            (table, child_column))
        if child.primary_key:
            raise CopyError(
                'a foreign key on the primary key is not supported: "%s.%s"'
                % (table, child_column))
        foreign_keys.append(ForeignKey(child_column, row[2], row[4]))

    return TableSchema(table, columns, primary_key, foreign_keys, create_sql)


def check_no_triggers(dataset, table, role):
    row = _fetchone(
        dataset,
        "SELECT name FROM sqlite_master "
        "WHERE type = 'trigger' AND lower(tbl_name) = lower(?) "
        "ORDER BY name LIMIT 1", (table,))
    if row is not None:
        raise CopyError(
            'table "%s" has trigger "%s" in the %s database; triggers are '
            'not supported' % (table, row[0], role))


def _check_compatible(source_schema, target_schema):
    if _lower(source_schema.primary_key.name) != \
            _lower(target_schema.primary_key.name):
        raise CopyError(
            'incompatible structure for "%s": primary key is "%s" in the '
            'source but "%s" in the target' % (
                source_schema.name, source_schema.primary_key.name,
                target_schema.primary_key.name))

    target_columns = target_schema.column_map
    for column in source_schema.columns:
        other = target_columns.get(_lower(column.name))
        if other is None:
            raise CopyError(
                'incompatible structure for "%s": target has no column "%s"'
                % (source_schema.name, column.name))
        if other.affinity != column.affinity:
            raise CopyError(
                'incompatible structure for "%s.%s": source type is %s but '
                'target type is %s' % (
                    source_schema.name, column.name, column.affinity,
                    other.affinity))

    # Extra target columns are fine only if SQLite can populate them.
    for column in target_schema.columns:
        if source_schema.column(column.name) is not None:
            continue
        if column.primary_key:
            raise CopyError(
                'incompatible structure for "%s": target primary key "%s" '
                'is missing from the source' % (
                    source_schema.name, column.name))
        if not column.nullable and not column.has_default:
            raise CopyError(
                'incompatible structure for "%s": target column "%s" is NOT '
                'NULL without a default' % (source_schema.name, column.name))

    # Foreign keys must match exactly.  An extra target key could reference
    # a record that is not part of this copy.
    source_fks = source_schema.fk_map
    target_fks = target_schema.fk_map
    for column_key, fk in source_fks.items():
        other = target_fks.get(column_key)
        if other is None:
            raise CopyError(
                'incompatible structure for "%s": source foreign key on '
                '"%s" is missing from the target' % (
                    source_schema.name, fk.column))
        if _lower(other.dest_table) != _lower(fk.dest_table):
            raise CopyError(
                'incompatible structure for "%s.%s": source references "%s" '
                'but target references "%s"' % (
                    source_schema.name, fk.column, fk.dest_table,
                    other.dest_table))
        if _lower(other.dest_column) != _lower(fk.dest_column):
            raise CopyError(
                'incompatible structure for "%s.%s": source references "%s" '
                'but target references "%s"' % (
                    source_schema.name, fk.column, fk.dest_column,
                    other.dest_column))
    for column_key, fk in target_fks.items():
        if column_key not in source_fks:
            raise CopyError(
                'incompatible structure for "%s": target has an extra '
                'foreign key on "%s"' % (source_schema.name, fk.column))


class _Structure:
    """Source/target schema pairs for one foreign-key closure."""

    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.source_schemas = {}
        self.target_schemas = {}

    def source_schema(self, table):
        key = _lower(table)
        if key not in self.source_schemas:
            self._load_closure(table)
        return self.source_schemas[key]

    def _load_closure(self, table):
        """Load every source schema reachable through foreign keys.

        Iterative, so a cyclic foreign-key graph produces a clean
        :class:`CopyError` later instead of a recursion error here.
        """
        stack = [table]
        while stack:
            name = stack.pop()
            key = _lower(name)
            if key in self.source_schemas:
                continue
            schema = load_table_schema(self.source, name)
            check_no_triggers(self.source, schema.name, 'source')
            self.source_schemas[key] = schema
            for fk in schema.foreign_keys:
                if _lower(fk.dest_table) not in self.source_schemas:
                    stack.append(fk.dest_table)

        # Resolve REFERENCES parent (no column) to the parent's primary key.
        for schema in self.source_schemas.values():
            for fk in schema.foreign_keys:
                parent = self.source_schemas.get(_lower(fk.dest_table))
                if parent is None:
                    raise CopyError(
                        'foreign key %s.%s references missing table "%s"' %
                        (schema.name, fk.column, fk.dest_table))
                dest_column = fk.dest_column or parent.primary_key.name
                if _lower(dest_column) != _lower(parent.primary_key.name):
                    raise CopyError(
                        'foreign key %s.%s must reference the primary key of '
                        '"%s"' % (schema.name, fk.column, parent.name))
                fk.dest_table = parent.name
                fk.dest_column = parent.primary_key.name

    def target_schema(self, table):
        key = _lower(table)
        if key not in self.target_schemas:
            source_schema = self.source_schema(table)
            target_name = _actual_table(self.target, source_schema.name)
            schema = load_table_schema(self.target, target_name)
            check_no_triggers(self.target, schema.name, 'target')
            for fk in schema.foreign_keys:
                try:
                    parent_name = _actual_table(self.target, fk.dest_table)
                except CopyError:
                    raise CopyError(
                        'incompatible structure for "%s": target foreign key '
                        'on "%s" references missing table "%s"' % (
                            source_schema.name, fk.column, fk.dest_table))
                if fk.dest_column is None:
                    parent = load_table_schema(self.target, parent_name)
                    fk.dest_column = parent.primary_key.name
                fk.dest_table = parent_name
            _check_compatible(source_schema, schema)
            self.target_schemas[key] = schema
        return self.target_schemas[key]

    def validate(self, table):
        """Validate the complete foreign-key closure of ``table``.

        Source and target structures are checked for every table that can
        be reached through foreign keys, before a single row is read.
        """
        self.source_schema(table)  # Loads the closure recursively.
        for name in sorted(self.source_schemas):
            self.target_schema(self.source_schemas[name].name)
        self.check_acyclic()

    def check_acyclic(self):
        state = {}

        def visit(table, path):
            status = state.get(table)
            if status == 'done':
                return
            if status == 'active':
                cycle = path[path.index(table):] + [table]
                raise CopyError('foreign key cycle detected: %s' %
                                ' -> '.join(
                                    self.source_schemas[name].name
                                    for name in cycle))
            state[table] = 'active'
            schema = self.source_schemas[table]
            for fk in sorted(schema.foreign_keys,
                             key=lambda fk: _lower(fk.dest_table)):
                visit(_lower(fk.dest_table), path + [table])
            state[table] = 'done'

        for table in sorted(self.source_schemas):
            visit(table, [])

    def insertion_order(self):
        order = []
        seen = set()

        def visit(table):
            if table in seen:
                return
            seen.add(table)
            for fk in sorted(self.source_schemas[table].foreign_keys,
                             key=lambda fk: _lower(fk.dest_table)):
                visit(_lower(fk.dest_table))
            order.append(table)

        for table in sorted(self.source_schemas):
            visit(table)
        return order


def _fetch_row(dataset, schema, key):
    columns = ', '.join(_quote(column.name) for column in schema.columns)
    row = _fetchone(
        dataset,
        'SELECT %s FROM %s WHERE %s = ?' % (
            columns, _quote(schema.name),
            _quote(schema.primary_key.name)),
        (key,))
    if row is None:
        raise CopyError('source row not found: %s primary key %s' %
                        (schema.name, key))
    return dict(zip([column.name for column in schema.columns], row))


def _resolve_rows(structure, table, keys):
    """Collect selected rows and every parent they reference (read-only)."""
    ordered = []
    seen = set()
    for key in keys:
        key = _check_integer_key(key, table, 'primary key')
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    if not ordered:
        raise CopyError('no rows were selected')

    root = structure.source_schema(table)
    root_key = _lower(root.name)
    rows = {}
    queue = [(root_key, key) for key in ordered]
    visited = set()

    while queue:
        table_key, key = queue.pop(0)
        if (table_key, key) in visited:
            continue
        visited.add((table_key, key))

        schema = structure.source_schemas[table_key]
        row = _fetch_row(structure.source, schema, key)
        rows.setdefault(table_key, {})[key] = row

        for fk in schema.foreign_keys:
            value = row[fk.column]
            if value is None:
                # NULL references stay NULL; nothing to fetch.
                continue
            parent = structure.source_schemas[_lower(fk.dest_table)]
            try:
                parent_key = _check_integer_key(
                    value, schema.name, fk.column)
                parent_row = _fetch_row(structure.source, parent, parent_key)
            except CopyError as exc:
                raise CopyError(
                    'missing referenced record: %s.%s = %r in "%s" (%s)' % (
                        schema.name, fk.column, value, parent.name, exc)
                ) from exc
            rows.setdefault(_lower(parent.name), {})[parent_key] = parent_row
            queue.append((_lower(parent.name), parent_key))
    return rows


def _prepare(source, target, table, keys):
    source_path = _database_path(source)
    target_path = _database_path(target)
    if source_path and target_path and \
            os.path.realpath(source_path) == os.path.realpath(target_path):
        raise CopyError('source and target databases must be different')
    structure = _Structure(source, target)
    structure.validate(table)
    rows = _resolve_rows(structure, table, keys)
    return structure, rows


def preflight_copy(source, target, table, keys):
    """Validate a copy and resolve every record that would be written.

    Raises :class:`CopyError` without modifying either database.  Returns
    the validated :class:`_Structure` and the rows that would be copied.
    """
    return _prepare(source, target, table, keys)


def _pragma_state(dataset, pragma):
    row = _fetchone(dataset, 'PRAGMA %s' % pragma)
    return bool(row[0]) if row else False


def _set_pragma(dataset, pragma, value):
    _query(dataset, 'PRAGMA %s = %s' % (pragma, 'ON' if value else 'OFF'))


def copy_referenced_records(source, target, table, keys):
    """Copy ``keys`` from ``table`` and every record they reference.

    ``source`` and ``target`` are DataSet-like objects exposing ``query``,
    ``transaction`` and ``filename``.  The source is only read.  The target
    is written in one transaction; any error rolls the whole batch back and
    raises :class:`CopyError`.  Existing target rows are never updated or
    replaced -- new primary keys are generated by the target.
    """
    structure, rows = _prepare(source, target, table, keys)
    order = structure.insertion_order()

    mapping = {}   # (table_key, source_key) -> new target key
    copied = {}    # table_key -> [(source_key, target_key)]
    old_foreign_keys = _pragma_state(target, 'foreign_keys')
    old_query_only = _pragma_state(source, 'query_only')
    # Belt and braces: the source connection must never write.
    _set_pragma(source, 'query_only', True)
    # Enforce references in the target so a bad rewrite cannot slip through.
    _set_pragma(target, 'foreign_keys', True)
    try:
        with target.transaction():
            for table_key in order:
                table_rows = rows.get(table_key)
                if not table_rows:
                    continue
                source_schema = structure.source_schemas[table_key]
                target_schema = structure.target_schemas[table_key]
                target_columns = target_schema.column_map

                insert_columns = []
                for column in source_schema.non_pk_columns:
                    other = target_columns.get(_lower(column.name))
                    if other is None:
                        # _check_compatible guarantees this cannot happen.
                        raise CopyError(
                            'incompatible structure for "%s": target has no '
                            'column "%s"' % (
                                source_schema.name, column.name))
                    insert_columns.append((column, other))

                sql = 'INSERT INTO %s (%s) VALUES (%s)' % (
                    _quote(target_schema.name),
                    ', '.join(_quote(other.name)
                              for _, other in insert_columns),
                    ', '.join('?' for _ in insert_columns))

                fk_by_column = {_lower(fk.column): fk
                                for fk in source_schema.foreign_keys}
                copied[table_key] = []
                for source_key in sorted(table_rows):
                    row = table_rows[source_key]
                    values = []
                    for column, _ in insert_columns:
                        value = row[column.name]
                        fk = fk_by_column.get(_lower(column.name))
                        if fk is not None and value is not None:
                            parent_key = _check_integer_key(
                                value, source_schema.name, column.name)
                            identity = (_lower(fk.dest_table), parent_key)
                            if identity not in mapping:
                                raise CopyError(
                                    'referenced record was not copied: '
                                    '%s.%s = %s' % (
                                        source_schema.name, column.name,
                                        value))
                            value = mapping[identity]
                        values.append(value)

                    cursor = _query(target, sql, tuple(values))
                    try:
                        new_key = cursor.lastrowid
                    finally:
                        cursor.close()
                    if new_key is None:
                        raise CopyError(
                            'target did not generate a primary key for "%s"'
                            % source_schema.name)
                    new_key = _check_integer_key(
                        new_key, target_schema.name,
                        target_schema.primary_key.name)
                    mapping[(table_key, source_key)] = new_key
                    copied[table_key].append((source_key, new_key))
    except CopyError:
        raise
    except Exception as exc:
        raise CopyError(
            'copy failed and the target was rolled back: %s' % exc) from exc
    finally:
        try:
            _set_pragma(target, 'foreign_keys', old_foreign_keys)
        except Exception:
            pass
        try:
            _set_pragma(source, 'query_only', old_query_only)
        except Exception:
            pass

    result = CopyResult(
        getattr(source, 'filename', '') or '',
        getattr(target, 'filename', '') or '')
    for table_key in order:
        if table_key in copied:
            schema = structure.source_schemas[table_key]
            result.tables.append(
                TableCopy(schema.name, len(copied[table_key]),
                          copied[table_key]))
    return result


def _decode_token(value):
    """Accept a plain integer or a row-key token produced by the UI."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    # Deferred import keeps this module usable on its own.
    try:
        try:
            from sqlite_web.executor import key_decode
        except ImportError:
            from executor import key_decode
        decoded = key_decode(str(value))
    except Exception:
        raise CopyError('invalid primary key: %r' % (value,))
    if len(decoded) != 1:
        raise CopyError('a single-column primary key was expected: %r' %
                        (value,))
    return decoded[0]


def parse_keys(text):
    """Parse a selection of integer primary keys from a form value.

    Accepts a JSON array, comma/whitespace separated values, repeated form
    values (passed as a list) or UI row-key tokens.
    """
    if text is None:
        values = []
    elif isinstance(text, (list, tuple)):
        values = []
        for entry in text:
            if isinstance(entry, str):
                entry = entry.strip()
                if not entry:
                    continue
                if entry.startswith('['):
                    values.extend(parse_keys(entry))
                    continue
                # The text field in the same form carries a comma/space
                # separated list.
                values.extend(entry.replace(',', ' ').replace(
                    '\r', ' ').replace('\n', ' ').replace('\t', ' ').split())
            else:
                values.append(entry)
    elif isinstance(text, str):
        stripped = text.strip()
        if not stripped:
            values = []
        elif stripped.startswith('['):
            try:
                values = json.loads(stripped)
            except ValueError as exc:
                raise CopyError('invalid key list: %s' % exc)
            if not isinstance(values, list):
                raise CopyError('key list must be a JSON array')
        else:
            values = stripped.replace(',', ' ').replace('\r', ' ').replace(
                '\n', ' ').replace('\t', ' ').split()
    else:
        values = [text]

    keys = []
    for value in values:
        if isinstance(value, str):
            value = value.strip()
            if len(value) > 1 and value[0] == value[-1] and \
                    value[0] in ('"', "'"):
                value = value[1:-1]
        keys.append(_check_integer_key(
            _decode_token(value), 'selection', 'primary key'))
    if not keys:
        raise CopyError('select at least one row to copy')
    return keys


# Short aliases, convenient for callers and tests.
copy_records = copy_referenced_records

#!/usr/bin/env python3
"""
SQLite数据库表结构查看工具
只显示表结构信息，不显示数据
"""
import sqlite3
import sys
from pathlib import Path

# 数据库文件路径
DB_FILE = Path(__file__).parent.parent / "resources" / "data" / "vitoom.db"


def get_table_schema(conn, table_name):
    """获取表的完整结构信息"""
    cursor = conn.cursor()
    
    # 获取列信息
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    # 获取索引信息
    cursor.execute(f"PRAGMA index_list({table_name})")
    indexes = cursor.fetchall()
    
    # 获取外键信息
    cursor.execute(f"PRAGMA foreign_key_list({table_name})")
    foreign_keys = cursor.fetchall()
    
    # 获取创建表的SQL语句
    cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    create_sql = cursor.fetchone()
    
    return {
        'columns': columns,
        'indexes': indexes,
        'foreign_keys': foreign_keys,
        'create_sql': create_sql[0] if create_sql else None
    }


def format_column_info(col):
    """格式化列信息"""
    col_id, name, col_type, not_null, default_val, pk = col
    
    parts = [f"  {name:<25} {col_type:<20}"]
    
    if pk:
        parts.append("PRIMARY KEY")
    if not_null:
        parts.append("NOT NULL")
    if default_val:
        parts.append(f"DEFAULT {default_val}")
    
    return " ".join(parts)


def show_table_schema(conn, table_name):
    """显示单个表的结构"""
    schema = get_table_schema(conn, table_name)
    
    print(f"\n{'='*80}")
    print(f"表名: {table_name}")
    print(f"{'='*80}")
    
    # 显示列信息
    print("\n📋 列信息:")
    print("-" * 80)
    print(f"{'列名':<25} {'类型':<20} {'约束'}")
    print("-" * 80)
    
    for col in schema['columns']:
        col_id, name, col_type, not_null, default_val, pk = col
        constraints = []
        if pk:
            constraints.append("PRIMARY KEY")
        if not_null:
            constraints.append("NOT NULL")
        if default_val:
            constraints.append(f"DEFAULT {default_val}")
        
        constraint_str = ", ".join(constraints) if constraints else "-"
        print(f"{name:<25} {col_type:<20} {constraint_str}")
    
    # 显示索引信息
    if schema['indexes']:
        print("\n🔍 索引信息:")
        print("-" * 80)
        cursor = conn.cursor()
        for idx in schema['indexes']:
            idx_name = idx[1]
            unique = "UNIQUE" if idx[2] else ""
            cursor.execute(f"PRAGMA index_info({idx_name})")
            idx_info = cursor.fetchall()
            cols = [col[2] for col in idx_info]
            print(f"  {idx_name:<30} {unique} ({', '.join(cols)})")
    
    # 显示外键信息
    if schema['foreign_keys']:
        print("\n🔗 外键信息:")
        print("-" * 80)
        for fk in schema['foreign_keys']:
            fk_id, seq, table, from_col, to_col, on_update, on_delete, match = fk
            print(f"  {from_col} -> {table}.{to_col}")
            if on_update != "NO ACTION":
                print(f"    ON UPDATE {on_update}")
            if on_delete != "NO ACTION":
                print(f"    ON DELETE {on_delete}")
    
    # 显示创建表的SQL
    if schema['create_sql']:
        print("\n📝 创建表的SQL:")
        print("-" * 80)
        # 格式化SQL，添加缩进
        sql_lines = schema['create_sql'].split('\n')
        for line in sql_lines:
            if line.strip():
                print(f"  {line}")
    
    print()


def list_all_tables(conn):
    """列出所有表"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    return tables


def show_all_schemas(conn):
    """显示所有表的结构"""
    tables = list_all_tables(conn)
    
    print(f"\n📊 数据库: {DB_FILE.name}")
    print(f"📋 共 {len(tables)} 个表")
    
    for table_name in tables:
        show_table_schema(conn, table_name)


def show_summary(conn):
    """显示数据库概览"""
    tables = list_all_tables(conn)
    
    print(f"\n{'='*80}")
    print(f"数据库概览: {DB_FILE.name}")
    print(f"{'='*80}")
    print(f"\n📊 表列表 (共 {len(tables)} 个):")
    print("-" * 80)
    
    cursor = conn.cursor()
    for table_name in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_count = len(columns)
        
        print(f"  {table_name:<30} {col_count:>3} 列, {count:>6} 条记录")
    
    print()


def main():
    if not DB_FILE.exists():
        print(f"❌ 数据库文件不存在: {DB_FILE}")
        sys.exit(1)
    
    try:
        conn = sqlite3.connect(str(DB_FILE))
        
        if len(sys.argv) > 1:
            table_name = sys.argv[1]
            
            # 检查是否是特殊命令
            if table_name == "--summary" or table_name == "-s":
                show_summary(conn)
                return
            
            # 检查表是否存在
            tables = list_all_tables(conn)
            if table_name not in tables:
                print(f"❌ 表 '{table_name}' 不存在")
                print(f"\n可用表列表:")
                for t in tables:
                    print(f"  • {t}")
                sys.exit(1)
            
            # 显示指定表的结构
            show_table_schema(conn, table_name)
        else:
            # 显示所有表的结构
            show_all_schemas(conn)
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


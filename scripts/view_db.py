#!/usr/bin/env python3
"""
SQLite数据库查看工具
快速查看数据库表结构和数据
"""
import sqlite3
import sys
from pathlib import Path

# 数据库文件路径
DB_FILE = Path(__file__).parent.parent / "resources" / "data" / "vitoom.db"


def show_tables(conn):
    """显示所有表名"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    print("\n📊 数据库表列表:")
    print("-" * 50)
    for table in tables:
        print(f"  • {table[0]}")
    print()
    return [table[0] for table in tables]


def show_table_schema(conn, table_name):
    """显示表结构"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    print(f"\n📋 表结构: {table_name}")
    print("=" * 80)
    
    headers = ["列名", "类型", "非空", "默认值", "主键"]
    rows = []
    for col in columns:
        rows.append([
            col[1],  # 列名
            col[2],  # 类型
            "是" if col[3] else "否",  # 非空
            col[4] if col[4] else "",  # 默认值
            "是" if col[5] else "否"   # 主键
        ])
    
    # 打印表头
    print(" | ".join(f"{h:15}" for h in headers))
    print("-" * 80)
    # 打印数据行
    for row in rows:
        print(" | ".join(f"{str(cell):15}" for cell in row))
    print()


def show_table_data(conn, table_name, limit=10):
    """显示表数据（限制行数）"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total = cursor.fetchone()[0]
    
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
    rows = cursor.fetchall()
    
    if not rows:
        print(f"\n📭 表 {table_name} 为空")
        return
    
    # 获取列名
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]
    
    print(f"\n📄 表数据: {table_name} (显示前 {min(limit, total)} 条，共 {total} 条)")
    print("=" * 80)
    
    # 格式化数据
    formatted_rows = []
    for row in rows:
        formatted_row = []
        for val in row:
            if val is None:
                formatted_row.append("NULL")
            elif isinstance(val, bytes):
                formatted_row.append(f"<binary {len(val)} bytes>")
            elif isinstance(val, str) and len(val) > 50:
                formatted_row.append(val[:47] + "...")
            else:
                formatted_row.append(str(val))
        formatted_rows.append(formatted_row)
    
    # 打印表头
    col_widths = [max(len(str(col)), max((len(str(row[i])) for row in formatted_rows), default=0)) for i, col in enumerate(columns)]
    print(" | ".join(f"{col:{col_widths[i]}}" for i, col in enumerate(columns)))
    print("-" * sum(col_widths + [len(col_widths) * 3 - 3]))
    # 打印数据行
    for row in formatted_rows:
        print(" | ".join(f"{str(cell):{col_widths[i]}}" for i, cell in enumerate(row)))
    print()


def show_indexes(conn, table_name):
    """显示表的索引"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA index_list({table_name})")
    indexes = cursor.fetchall()
    
    if indexes:
        print(f"\n🔍 索引信息: {table_name}")
        print("-" * 50)
        for idx in indexes:
            idx_name = idx[1]
            cursor.execute(f"PRAGMA index_info({idx_name})")
            idx_info = cursor.fetchall()
            cols = [col[2] for col in idx_info]
            print(f"  索引: {idx_name} -> {', '.join(cols)}")
        print()


def main():
    if not DB_FILE.exists():
        print(f"❌ 数据库文件不存在: {DB_FILE}")
        sys.exit(1)
    
    try:
        conn = sqlite3.connect(str(DB_FILE))
        
        # 显示所有表
        tables = show_tables(conn)
        
        if len(sys.argv) > 1:
            # 如果指定了表名，只显示该表的信息
            table_name = sys.argv[1]
            if table_name not in tables:
                print(f"❌ 表 '{table_name}' 不存在")
                sys.exit(1)
            
            show_table_schema(conn, table_name)
            show_table_data(conn, table_name, limit=20)
            show_indexes(conn, table_name)
        else:
            # 显示所有表的结构和数据
            for table_name in tables:
                show_table_schema(conn, table_name)
                show_table_data(conn, table_name, limit=5)
                show_indexes(conn, table_name)
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


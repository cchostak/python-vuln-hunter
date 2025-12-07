"""Safe SQLite usage with parameterized queries."""
import sqlite3
from contextlib import closing


def get_user(conn, username):
    with closing(conn.cursor()) as cur:
        cur.execute("SELECT id, username FROM users WHERE username = ?", (username,))
        return cur.fetchone()


def create_user(conn, username, password_hash):
    with closing(conn.cursor()) as cur:
        cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()


def init_db(path: str = ":memory:"):
    conn = sqlite3.connect(path)
    with closing(conn.cursor()) as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT)")
    return conn


if __name__ == "__main__":
    conn = init_db()
    create_user(conn, "alice", "hash")
    print(get_user(conn, "alice"))

"""Simple safe math utilities."""

def add(a, b):
    return a + b

def average(items):
    if not items:
        return 0
    return sum(items) / len(items)

if __name__ == "__main__":
    print(add(2, 3))
    print(average([1, 2, 3]))

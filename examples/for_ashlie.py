n = 2

count = 0
for j in range(1, n+1):
    count = count + 1
    print("outer")
    for i in range(1, j+1):
        count = count + 1
        print('inner')
print(f"Count is: {count} when m is {n}")

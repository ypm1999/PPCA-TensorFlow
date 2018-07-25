from Queue import PriorityQueue

n = input()
Q = PriorityQueue()
Q.put(-input())
ans = 0
for i in range(1, n):
    x = -input()
    y = Q.get()
    if -y > -x:
        ans += -y + x
        y = x
    Q.put(x)
    Q.put(y)
print(ans)

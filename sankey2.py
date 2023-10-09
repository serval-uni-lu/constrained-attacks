success_rate = [
    0.0,
    0.16997167138810199,
    0.43342776203966005,
    0.8059490084985835,
]

fail_rate = [1 - e for e in success_rate]


delta_success = []
for i in range(len(success_rate) - 1):
    delta_success.append(success_rate[i + 1] - success_rate[i])

delta_fail = []
for i in range(len(success_rate) - 1):
    delta_fail.append(1 - success_rate[i + 1] - (1 - success_rate[i]))


for e in delta_success:
    print(f"{e:.3f}")
    
for e in fail_rate:
    print(f"{e:.3f}")
    
    
    
X_clean [83.0] Fail1 #b7472a
X_clean [17.0] Success1 #217346

Fail1 [56.7] Fail2 #b7472a
Fail1 [26.3] Success2 #217346

Fail2 [19.4] Fail3 #b7472a
Fail3 [19.4] Fail4 #b7472a
Fail2 [37.3] Success3 #217346
Success3 [37.3] Success4 #217346

Success2 [26.3] Success4 #217346
Success1 [17.0] Success4 #217346


0 [83.0] 1 #b7472a
0 [17.0] 2 #217346

1 [56.7] 3 #b7472a
1 [26.3] 4 #217346

3 [19.4] 5 #b7472a
5 [19.4] 7 #b7472a
3 [37.3] 6 #217346
6 [37.3] 8 #217346

4 [26.3] 8 #217346
2 [17.0] 8 #217346


    
X_clean [83.0] F1 #b7472a
X_clean [17.0] S1 #217346

F1 [56.7] F2 #b7472a
F1 [26.3] S2 #217346

F2 [19.4] F3 #b7472a
F3 [19.4] F4 #b7472a
F2 [37.3] S3 #217346
S3 [37.3] S4 #217346

S2 [26.3] S4 #217346
S1 [17.0] S4 #217346
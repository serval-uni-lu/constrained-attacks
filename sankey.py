import numpy as np

fail_color = "#b7472a"
success_color = "#217346"
steps = ["CPGD", "CAPGD", "MOEVA"]
fail = [0.706, 0.584, 0.477, 0.0150]
fail = np.array(fail)
fail = fail / fail[0]
fail = np.round(fail, 3)
fails = fail[1:]
successes = []
for i in range(len(fails)):
    if i == 0:
        successes.append(1 - fails[i])
    else:
        successes.append(fails[i - 1] - fails[i])
successes = np.array(successes)

# print(fail)

# for i in range(len(steps) - 1):
#     print(f"{steps[i]} [{fail[i]:.3f}] {steps[i+1]} {fail_color}")

# # print(f"{steps[-1]} [{fail[-1]:.3f}] X_out_fail {fail_color}")

# for i in range(len(steps) - 1):
#     if i > 0:
#         success = fail[i - 1] - fail[i]
#         print(f"{steps[i]} [{success:.3f}] X_out_success {success_color}")


# for i in range(len(steps) - 3):
#     fail = fails[i]
#     print(f"{steps[i]}_F [{fail:.3f}] {steps[i+1]}_F {fail_color}")
#     success = fails[i] - fails[i + 1]
#     print(f"{steps[i]} [{success:.3f}] {steps[i+1]}_P {fail_color}")

start = "X_clean"
end_success = "X_out Success"
end_fail = "X_out Fail"
count = 1
name_to_index = {
    "start": 0,
}

for i, step in enumerate(steps):
    name_to_index[step] = {"F": count, "P": count + 1}
    count += 2
name_to_index[end_fail] = count
name_to_index[end_success] = count + 1
fail_success = {
    "Fail": fails,
    "Success": successes,
}

label = (
    [
        f"{start}: 1.00",
    ]
    + [
        f"{status}: {fail_success[status][i]:.3f}"
        for i, step in enumerate(steps)
        for status in ["Fail", "Success"]
    ]
    + [
        f"{end_fail}: {fails[-1]:.3f}",
        f"{end_success}: {1-fails[-1]:.3f}",
    ]
)
sources = []
targets = []
values = []
colors = (
    [fail_color]
    + [fail_color, success_color] * len(steps)
    + [fail_color, success_color] * 2
)


def add_edge(
    source,
    target,
    value,
):
    sources.append(source)
    targets.append(target)
    values.append(value)


for i in range(len(steps)):
    fail = fails[i]
    success = successes[i]

    # Fail path
    if i == 0:
        origin = name_to_index["start"]
    else:
        origin = name_to_index[steps[i - 1]]["F"]

    end = name_to_index[steps[i]]["F"]
    add_edge(origin, end, fail)

    # Success path
    if i == 0:
        origin = name_to_index["start"]
    else:
        origin = name_to_index[steps[i - 1]]["F"]
    end = name_to_index[steps[i]]["P"]
    add_edge(origin, end, success)

    # Success path to out
    origin = name_to_index[steps[i]]["P"]
    end = name_to_index[end_success]
    add_edge(origin, end, success)

    if i == len(steps) - 1:
        origin = name_to_index[steps[i]]["F"]
        end = name_to_index[end_fail]
        add_edge(origin, end, fail)


print("source = ", sources)
print("target = ", targets)
print("value = ", values)
print("label = ", label)
print("color = ", colors)

print(" NEW WAY -----")


for source, target, value, color in zip(sources, targets, values, colors):
    print(f"{source} [{value:.3f}] {target} {color}")
    

print(" OLD WAY -----")
for i in range(len(steps)):
    fail = fails[i]
    success = successes[i]

    # Fail path
    if i == 0:
        origin = start
    else:
        origin = f"{steps[i-1]}_F"
    print(f"{origin} [{fail:.3f}] {steps[i]}_F {fail_color}")

    # Success path
    if i == 0:
        origin = start
    else:
        origin = f"{steps[i-1]}_F"
    print(f"{origin} [{success:.3f}] {steps[i]}_P {success_color}")

    # Success path to out
    print(f"{steps[i]}_P [{success:.3f}] X_out_P {success_color}")
    # Fail path to out
    # print(f"{steps[i]}_F [{fail:.3f}] X_out_F {fail_color}")

    if i == len(steps) - 1:
        print(f"{steps[i]}_F [{fail:.3f}] X_out_F {fail_color}")


# X_clean [1] CPGD #b7472a
# CPGD [0.75] CAPGD #b7472a
# CAPGD [0.66] MOEVA #b7472a
# MOEVA [0.33] X_out #b7472a


# CAPGD [0.09] X_out #217346
# CPGD [0.25] X_out #217346
# MOEVA [0.33] X_out #217346



0 [0.83] 1 #b7472a
0 [0.17] 2 #217346

1 [0.68] 3 #b7472a
1 [0.15] 4 #217346

3 [0.02] 5 #b7472a
5 [0.02] 7 #b7472a
3 [0.66] 6 #217346
6 [0.66] 8 #217346

4 [0.15] 8 #217346
2 [0.17] 8 #217346



X_clean [0.827] CPGD_F #b7472a
X_clean [0.173] CPGD_P #217346
CPGD_P [0.173] X_out_P #217346
CPGD_F [0.676] CAPGD_F #b7472a
CPGD_F [0.151] CAPGD_P #217346
CAPGD_P [0.151] X_out_P #217346
CAPGD_F [0.021] MOEVA_F #b7472a
CAPGD_F [0.655] MOEVA_P #217346
MOEVA_P [0.655] X_out_P #217346
MOEVA_F [0.021] X_out_F #b7472a15.
from collections import defaultdict
import math

d_sim_ds_m = 1200  # Set the maximum matching distance
d_match_rate = 0.75   # Set the matching rate threshold

def match(trajs_ds1, trajs_ds2, pos_dim):
    # Initialize counters
    cnt = defaultdict(int)
    cnt2 = defaultdict(int)
    print('match_ds2ds')

    # 封装距离计算函数
    def dist(point1, point2):
        dist_squared = 0
        for i in range(pos_dim):
            dist_squared += (point1[i] - point2[i]) ** 2
        return math.sqrt(dist_squared) * 109001.77571  # Convert to meters

    # 封装相似度判断函数
    def cal_sim_ds(point1, point2):
        return dist(point1, point2) < d_sim_ds_m

    # Iterate through trajectories from both data source
    for id_traj, traj_ds1_data in trajs_ds1.items():
        for other_id_traj, traj_ds2_data in trajs_ds2.items():
            # Initialize matching pointers and limit condition
            p_ds, p2_ds = 0, 0
            lim = 0  # Controls consecutive mismatches
            # Compare trajectory points based on time and adjust pointers
            while p_ds < len(traj_ds1_data) and p2_ds < len(traj_ds2_data):
                def t_ds1(p):
                    return traj_ds1_data[p][0]

                def t_ds2(p):
                    return traj_ds2_data[p][0]

                while p_ds < len(traj_ds1_data) and t_ds1(p_ds) < t_ds2(p2_ds):
                    p_ds += 1
                if p_ds >= len(traj_ds1_data):
                    break
                while p2_ds + 1 < len(traj_ds2_data) and t_ds2(p2_ds + 1) <= t_ds1(p_ds):
                    p2_ds += 1
                if p2_ds >= len(traj_ds2_data):
                    break

                # Update matching counters; terminate if consecutive mismatches exceed 30
                cnt2[((id_traj), (other_id_traj))] += 1
                point1 = traj_ds1_data[p_ds][1:1 + pos_dim]
                point2 = traj_ds2_data[p2_ds][1:1 + pos_dim]
                if cal_sim_ds(point1, point2):
                    lim = 0
                    cnt[((id_traj), (other_id_traj))] += 1
                else:
                    lim += 1
                    if lim > 30:
                        break
                p2_ds += 1

    # Process matching results
    buff_del = []
    for label in cnt.keys():
        # Filter out trajectory pairs with low matching counts
        if cnt[label] < d_match_rate * cnt2[label]:
            buff_del.append(label)
    for i in buff_del:
        del cnt[i]

    # Categorize trajectories based on matching degree
    v_label = defaultdict(lambda: [])
    for label, v in cnt.items():
        v_label[v].append(label)

    # Generate final matching results
    disjoint = {}  # Dictionary to store matching between ds 1 and ds 2 trajectories
    dic_match = {}  # Store the final matches

    # Traverse the trajectory pairs sorted by matching degree
    for v, label_list in sorted(v_label.items(), key=lambda d: d[0], reverse=True):
        for (id_traj, other_id_traj) in label_list:
            # Skip matching if either trajectory ID is already used
            if id_traj in disjoint or other_id_traj in disjoint.values():
                continue

            # Add current pair to the disjoint dictionary to ensure uniqueness
            disjoint[id_traj] = other_id_traj

            # Save matching results to dic_match dictionary
            if id_traj not in dic_match:
                dic_match[id_traj] = []
            dic_match[id_traj].append(other_id_traj)

    return dic_match
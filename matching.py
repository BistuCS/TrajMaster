from collections import defaultdict
import math

d_sim_ds_m = 1200  # Set the maximum matching distance
d_match_rate = 0.75   # Set the matching rate threshold

def match(trajs_ds1, trajs_ds2, pos_dim):
    # Initialize counters
    cnt = defaultdict(int)
    cnt2 = defaultdict(int)
    print('match_ds2ds')

    # Iterate through trajectories from both data source
    for id_traj, traj_ds1_data in trajs_ds1.items():
        for other_id_traj, traj_ds2_data in trajs_ds2.items():
            # Initialize matching pointers and limit condition
            p_ds, p2_ds = [0] * 2
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

                # Calculate trajectory point similarity
                def dist2(alng, alat, blng, blat):
                    return math.sqrt((alng - blng) ** 2 + (alat - blat) ** 2) * 109001.77571  # Convert to meters

                def dist3(alng, alat, alti, blng, blat, blti):
                    return math.sqrt((alng - blng) ** 2 + (alat - blat) ** 2 + (alti - blti) ** 2) * 109001.77571  # Convert to meters

                def cal_sim_ds_2d(lon, lat, lon2, lat2):
                    return dist2(lon, lat, lon2, lat2) < d_sim_ds_m

                def cal_sim_ds_3d(lon, lat, z, lon2, lat2, z2):
                    return dist3(lon, lat, z, lon2, lat2, z2) < d_sim_ds_m

                # Update matching counters; terminate if consecutive mismatches exceed 30
                if pos_dim == 2:
                    cnt2[((id_traj), (other_id_traj))] += 1
                    if cal_sim_ds_2d(*traj_ds1_data[p_ds][1:3], *traj_ds2_data[p2_ds][1:3]):
                        lim = 0
                        cnt[((id_traj), (other_id_traj))] += 1
                    else:
                        lim += 1
                        if lim > 30:
                            break
                elif pos_dim == 3:
                    cnt2[((id_traj), (other_id_traj))] += 1
                    if cal_sim_ds_3d(*traj_ds1_data[p_ds][1:4], *traj_ds2_data[p2_ds][1:4]):
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

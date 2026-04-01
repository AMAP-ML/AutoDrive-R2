import json
import re
import os
import argparse
from collections import Counter
import re
from typing import List, Dict, Optional

# run example:
# python calculate_all.py  --pred_folder /mnt/private-user-data/ed/lm/vla/temp/eval/output_1.json \
#     --save_path ./result.json

parser = argparse.ArgumentParser(description='')
parser.add_argument('--pred_folder', type=str, required=True,
                    help='/path/to/pred/folder')
parser.add_argument('--save_path', type=str, required=True,
                    help='/path/to/save/results')

args = parser.parse_args()
      

class Accuracy_task:
    def __init__(self, type, pred, gt, formatted=True):
        self.type = type
        self.pred = pred
        self.formatted = formatted
        if self.formatted:
            self.pred = jsonalize(self.pred)
        self.gt = gt
        # self.not_match_count = 0

    def execute(self, idx):
        
        if self.type == 'q7':

            pattern = r'\[(-?\d+\.\d+),\s*(-?\d+\.\d+)\]'
            matches = re.findall(pattern, self.gt)
            # import pdb; pdb.set_trace()

            gt_coordinates = [(float(x), float(y)) for x, y in matches]
            if self.pred == {}:
                return None
            # print(self.pred)
            if isinstance(self.pred, str):
                pattern = r'\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]'
                # import pdb; pdb.set_trace()
                matches = re.findall(pattern, self.pred)
                self.pred = [(float(x), float(y)) for x, y in matches]
                # import pdb; pdb.set_trace()
                if len(self.pred )< 6:
                    # self.not_match_count +=1
                    print(idx, " len =================== ", len(self.pred))
                    return None
                # 
                try:
                    self.pred = {
                        'predicted_waypoints': {
                            't+0.5s': self.pred[0],
                            't+1.0s': self.pred[1],
                            't+1.5s': self.pred[2],
                            't+2.0s': self.pred[3],
                            't+2.5s': self.pred[4],
                            't+3.0s': self.pred[5] 
                        }
                    }
                except:
                    import pdb; pdb.set_trace()
                    self.pred = {
                        'predicted_waypoints': {
                            't+0.5s': [0, 0],
                            't+1.0s': [0, 0],
                            't+1.5s': [0, 0],
                            't+2.0s': [0, 0],
                            't+2.5s': [0, 0],
                            't+3.0s': [0, 0],
                            't+3.5s': [0, 0],
                            't+4.0s': [0, 0],
                            't+4.5s': [0, 0],
                            't+5.0s': [0, 0]
                        }
                    }
                    
            pred_coordinates = [
                self.pred['predicted_waypoints']['t+0.5s'],
                self.pred['predicted_waypoints']['t+1.0s'],
                self.pred['predicted_waypoints']['t+1.5s'],
                self.pred['predicted_waypoints']['t+2.0s'],
                self.pred['predicted_waypoints']['t+2.5s'],
                self.pred['predicted_waypoints']['t+3.0s'],
                # self.pred['predicted_waypoints']['t+3.5s'],
                # self.pred['predicted_waypoints']['t+4.0s'],
                # self.pred['predicted_waypoints']['t+4.5s'],
                # self.pred['predicted_waypoints']['t+5.0s'],
            ]
            l2_loss = 0
            loss_batch = []

            # vad patten
            # 0.5 s
            l2_0_5 = ((gt_coordinates[0][0]-pred_coordinates[0][0]) **
                    2+(gt_coordinates[0][1]-pred_coordinates[0][1])**2)**0.5
            # 1.0 s
            l2_1_0 = ((gt_coordinates[1][0]-pred_coordinates[1][0]) **
                    2+(gt_coordinates[1][1]-pred_coordinates[1][1])**2)**0.5
            l2_1_0 = (l2_0_5 + l2_1_0 )/2
            # 1.5 s
            l2_1_5 = ((gt_coordinates[2][0]-pred_coordinates[2][0]) **
                    2+(gt_coordinates[2][1]-pred_coordinates[2][1])**2)**0.5
            l2_1_5 = (l2_0_5 + l2_1_0 + l2_1_5)/3
            # 2.0 s
            l2_2_0 = ((gt_coordinates[3][0]-pred_coordinates[3][0]) **
                    2+(gt_coordinates[3][1]-pred_coordinates[3][1])**2)**0.5
            l2_2_0 = (l2_0_5 + l2_1_0 + l2_1_5 + l2_2_0)/4
            # 2.5 s
            l2_2_5 = ((gt_coordinates[4][0]-pred_coordinates[4][0]) **
                    2+(gt_coordinates[4][1]-pred_coordinates[4][1])**2)**0.5
            l2_2_5 = (l2_0_5 + l2_1_0 + l2_1_5 + l2_2_0 + l2_2_5)/5
            # 3.0 s
            l2_3_0 = ((gt_coordinates[5][0]-pred_coordinates[5][0]) **
                    2+(gt_coordinates[5][1]-pred_coordinates[5][1])**2)**0.5
            l2_3_0 = (l2_0_5 + l2_1_0 + l2_1_5 + l2_2_0 + l2_2_5 + l2_3_0)/6

            loss_batch.append((l2_0_5 + l2_1_0)/2)
            loss_batch.append((l2_0_5 + l2_1_0 + l2_1_5 +l2_2_0)/4)
            loss_batch.append((l2_0_5 + l2_1_0 + l2_1_5 +l2_2_0 + l2_2_5 + l2_3_0)/6)
            return loss_batch

def get_sorted_paths(dirname):
    filelist = os.listdir(dirname)
    filelist.sort()
    filelist = [os.path.join(dirname, file)
                for file in filelist if '.json' in file]
    return filelist

def jsonalize(text):
    try:
        text = json.loads(text)
        return text
    except:
        pass
    try:
        text = text.split("```json\n")[1].split("\n```")[0]
        text = json.loads(text)
        return text
    except:
        return text


results_all = []
mode = 'sum'  # sum or detail
not_match_count =0
if True:

    pred_path = args.pred_folder #"/nuscenes_test.json"
    q = 'q7'
    print(f"Processing {q} {pred_path} ...")
    
    with open(pred_path, 'r') as f:
        pred_data = json.load(f) #[json.loads(line) for line in f]
    gt_data = pred_data
    print(len(pred_data), len(gt_data))
    is_format = False
    tasks = []

    size_gt = len(gt_data)
    for i in range(size_gt):
        if gt_data[i]["problem_id"] == pred_data[i]["problem_id"]:

            tasks.append(Accuracy_task(q, pred_data[i]['predict'], gt_data[i]['solution'], is_format))

    if True:

        if mode == 'sum' :
            all_loss_1 = 0
            all_loss_2 = 0
            all_loss_3 = 0
            # all_loss_4 = 0
            # all_loss_5 = 0
            all_cnt = 0
            idx = 0
            for task in tasks:
                # import pdb; pdb.set_trace()
                
                loss = task.execute(idx)
                idx+=1
                if loss == None:
                    not_match_count +=1
                    
                    continue
                else:
                    if loss is not None:
                        all_loss_1 += loss[0]
                        all_loss_2 += loss[1]
                        all_loss_3 += loss[2]
                        # all_loss_4 += loss[3]
                        # all_loss_5 += loss[4]

                    all_cnt += 1
            results_all.append({'q': q, 'loss_1': all_loss_1/all_cnt, 'loss_2': all_loss_2/all_cnt, 'loss_3': all_loss_3 /
                               all_cnt,  'pred_path': pred_path, 'cnt': all_cnt})
        else:
            results = [task.execute() for task in tasks]
            results_all.append(
                {'q': q, 'filepath': pred_path, 'cnt': len(tasks), 'results': results})
results_all[0]["avg"] = (results_all[0]["loss_1"]+results_all[0]["loss_2"]+results_all[0]["loss_3"])/3
print(" result =", results_all)
print(" not_match_count = ", not_match_count)
with open(args.save_path, 'w') as f:
    json.dump(results_all, f, indent=4)

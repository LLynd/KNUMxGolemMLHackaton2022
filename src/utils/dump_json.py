import json

json_path = '../../data/images_part2_test_public_copy.json'

with open(json_path, 'r+') as f:
    data = json.load(f)
    for i in range(len(results_df)):
        if results_df.iloc[i]['preds'] == results_df.iloc[i]['preds']:
            data['annotations'][i]['category_id'] = results_df.iloc[i]['preds']
        else:
            data['annotations'][i]['category_id'] = None
    f.seek(0)        # <--- should reset file position to the beginning.
    json.dump(data, f, indent=4)
    f.truncate()     # remove remaining part

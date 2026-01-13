import json
from pathlib import Path
import os
import pandas as pd
import zipfile
from datetime import datetime
import shutil
import glob
from soilfauna.config import default
from soilfauna.logging import LOGGER


def merge_list(x):
        res = list()
        for i in x:
            res.extend(i)
        return res

def coco2df(coco):
    '''
    Fit a coco instance into a flat pandas DataFrame.
    '''
    classes_df = pd.DataFrame(coco['categories'])
    classes_df['name'] = classes_df['name'].str.strip()
    classes_df = classes_df.rename(columns={"id": "category_id"})
    images_df = pd.DataFrame(coco['images'])
    images_df.rename(columns={"id": "image_id"}, inplace=True)
    coco_df = pd.DataFrame(coco['annotations'])\
                    .merge(classes_df, on="category_id", how='left')\
                    .merge(images_df, on="image_id", how='left')

    return coco_df

def convert(
        coco_file: str | Path,
        label_tree_path: str | Path,
        name: str | None = None,
        output_dir: str | Path = default.DEFAULT_COCO2BIIGLE_OUTPUT_DIR,
        project_name: str = "project01",
        volume_name: str = "volume01"
    ):

    with open(coco_file, 'r') as f:
        coco = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not name:
        name = str(len(next(os.walk(output_dir))[1]))

    output_path = Path(os.path.join(output_dir, name, project_name, volume_name))
    temp_path = Path(os.path.join(output_dir, name, 'temp'))

    output_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)

    df = coco2df(coco)

    df['annotation_id'] = df.index + 1
    
    if not label_tree_path.exists():
        raise FileNotFoundError(f"{label_tree_path.absolute()} does not exist")

    with zipfile.ZipFile(label_tree_path, 'r') as zip:
        zip.extractall(temp_path)

    label_tree_path = os.path.join(temp_path, 'label_trees.json')
    user_path = os.path.join(temp_path, 'users.json')

    with open(label_tree_path, 'r') as f:
        label_tree = json.load(f)

    labels = label_tree[0].get('labels')

    labels_map = {label['name']: label['id'] for label in labels}

    df['label_id'] = df['name'].map(labels_map)

    with open(user_path, 'r') as f:
        users = json.load(f)

    df['user_id'] = users[0]['id']
    df['confidence'] = 1

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    df['created_at'] = now
    df['updated_at'] = now

    image_annotation_labels_cols = [
        'annotation_id',
        'label_id', 
        'user_id',
        'confidence',
        'created_at',
        'updated_at'
    ]

    # Dump image_annotation_labels.csv
    df[image_annotation_labels_cols].to_csv(
        os.path.join(output_path, "image_annotation_labels.csv"),
        sep=",",
        index=False,
        quotechar='"',
        quoting=2
    )

    # Dump image_annotations.csv
    df['id'] = df['annotation_id']
    df['shape_id'] = 3

    df['points'] = df['segmentation'].apply(merge_list)

    select_col = [
        'id',
        'image_id', 
        'shape_id',
        'created_at',
        'updated_at',
        'points'
    ]

    df[select_col].to_csv(
        os.path.join(output_path, "image_annotations.csv"),
        sep=",",
        index=False,
        quotechar='"',
        quoting=2
    )

    # Preparing image.csv
    df['id'] = df['image_id']
    df['filename'] = df['file_name']  # Needs to be updated with biigle image id
    df['volume_id'] = 1  # Need to be fixed ?

    select_col = [
        'id',
        'filename',
        'volume_id'
    ]

    df[select_col].drop_duplicates(keep='first').to_csv(
        os.path.join(output_path, "images.csv"),
        sep=",",
        index=False
    )

    volume = [{
        "id": 1,
        "name": volume_name,
        "url": f"local://{os.path.basename(volume_name)}",
        "attrs": None,
        "media_type_name": "image"
    }]

    with open(os.path.join(output_path, 'volumes.json'), "w") as js:
        json.dump(volume, js)

    for z in glob.glob(os.path.join(default.BIIGLE_MODEL_FILES_DIR, '*.csv')):
        shutil.copy(z, os.path.join(output_path, os.path.basename(z)))

    for z in glob.glob(os.path.join(temp_path, '*.json')):
        shutil.copy(z, os.path.join(output_path, os.path.basename(z)))

    to_compress = [f for f in glob.glob(os.path.join(output_path, '*'))]

    with zipfile.ZipFile(f"{os.path.join(output_path)}.zip", "w") as archive:
        for file in to_compress:
            archive.write(file, os.path.basename(file))
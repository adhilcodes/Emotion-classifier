import os
import pandas as pd

def prepare_data(data_path, sub_data):
    rooms = []
    for item in data_path:
        all_rooms = os.listdir(os.path.join(sub_data, item))

        for room in all_rooms:
            rooms.append((item, os.path.join(sub_data, item, room)))

    data = pd.DataFrame(data=rooms, columns=['tag', 'image'])
    return data

def label_mapping(data):
    df = data.loc[:, ['image', 'tag']]
    label_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }
    df['label'] = df['tag'].map(label_map)
    return df

def main():
    train_path = './data/train'
    test_path = './data/final test'
    val_path = './data/validation'

    train_data_path = os.listdir(train_path)
    test_data_path = os.listdir(test_path)
    val_data_path = os.listdir(val_path)

    train_data = prepare_data(train_data_path, train_path).loc[:, ['image', 'tag']]
    test_data = prepare_data(test_data_path, test_path)
    val_data = prepare_data(val_data_path, val_path)

    train_df = label_mapping(train_data)
    test_df = label_mapping(test_data)
    val_df = label_mapping(val_data)

    train_df.to_csv('./data/train.csv', index=False)
    test_df.to_csv('./data/test.csv', index=False)
    val_df.to_csv('./data/val.csv', index=False)

if __name__ == "__main__":
    main()


import json
import numpy as np
import tensorflow as tf


dataset_cache = None


def get_dataset():
    global dataset_cache
    if dataset_cache is None:
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        flattened_images = tf.reshape(train_images, [train_images.shape[0], -1])
        dataset_cache = {
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
            "flattened_images": flattened_images,
        }
    return dataset_cache


def knn(vector, k):
    data = get_dataset()
    train_labels = data["train_labels"]
    flattened_images = data["flattened_images"]

    diffs = flattened_images - vector
    sq_diffs = tf.square(diffs)
    sums = tf.reduce_sum(sq_diffs, axis=1)
    distances = tf.sqrt(sums)
    sorted_dist = tf.argsort(distances)

    nearest_labels = [int(train_labels[i]) for i in sorted_dist.numpy()[:k]]
    result = np.bincount(nearest_labels).argmax()
    return result, nearest_labels


def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        vector = tf.constant(body["vector"], dtype=tf.float32)
        k = int(body.get("k", 5))
        result, nearest_labels = knn(vector, k)
        return {
            "statusCode": 200,
            "body": json.dumps({"result": result, "nearestNeighbors": nearest_labels}),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

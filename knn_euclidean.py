import json
import numpy as np
import tensorflow as tf



def knn(vector, k):
    diffs = flattened_images - vector
    sq_diffs = tf.square(diffs)
    sums = tf.reduce_sum(sq_diffs, axis=1)
    distances = tf.sqrt(sums)
    sorted_dist = tf.argsort(distances)

    nearest_labels = [int(train_labels[i]) for i in sorted_dist.numpy()[:k]]
    most_common = np.bincount(nearest_labels).argmax()
    return most_common, nearest_labels


def lambda_controller(event, context):
    try:
        body = json.loads(event['body'])
        vector = tf.constant(body['vector'], dtype=tf.float32)
        k = int(body.get('k', 5))
        result, nearest_labels = knn(vector, k)
        return {
            'statusCode': 200,
            'body': json.dumps({'result': result, 'nearestNeighbors': nearest_labels})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

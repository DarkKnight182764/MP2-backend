# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import cv2
import numpy as np
from tensorflow import keras
from keras_vggface.utils import preprocess_input

print(tf.executing_eagerly())
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())
# exit()
# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
session = tf.compat.v1.Session(graph=tf.Graph())
with session.graph.as_default():
    set_session(session)
    import keras_vggface
    # print version
    print(keras_vggface.__version__)
    from keras_vggface.vggface import VGGFace
    # create a vggface model
    vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    print("Vgg face model loaded\n\n", vgg_model.summary())

    from mtcnn import MTCNN
    detector = MTCNN()
    print("mtcnn loaded\n\n")

    import dill
    with open("vocab_dump", "rb") as f:
        vocab = dill.load(f)

    model4 = keras.models.load_model("double_attention_250e_0.12_wer")
    print("lip reading model loaded\n\n")


def _get_sent_from_preds(labels):  # (bs, 75, 53)
    global vocab
    labels = np.argmax(labels, axis=2)  # (bs, 75)
    # print(labels.shape)
    res = []
    for vid_label in labels:
        vid_res = ["sil"]
        for word_index in vid_label:
            word = vocab[word_index]
            if not vid_res[-1] == word:
                vid_res.append(word)
        res.append(" ".join(vid_res).strip())
    return res


def predict(filepath):
    global model4, detector, vocab, vgg_model, session
    with session.graph.as_default():
        set_session(session)
        X = np.zeros((1, 75, 2048), dtype=np.float32)
        cam = cv2.VideoCapture(filepath)
        frame_buffer = np.zeros((75, 224, 224, 3), dtype=np.float32)
        for num_frame in range(75):
            ret, frame = cam.read()
            if frame is not None:
                frame = frame.astype(np.float32)
                # print("before detect")
                results = detector.detect_faces(frame)
                # print("after detect")
                if len(results) >= 1:
                    x1, y1, width, height = results[0]['box']
                    x2, y2 = x1 + width, y1 + height
                    frame = frame[y1:y2, x1:x2]
                frame = cv2.resize(frame, (224, 224)).astype(np.float32)
                frame_buffer[num_frame] = frame

        print("Lip region cropping done")
        frame_buffer = preprocess_input(frame_buffer)
        X[0] = vgg_model.predict(frame_buffer)
        print("VGG model predict done")
        pred = model4.predict(X)[1]
        print("LSTM model predict done")
        # print(pred.shape)
        return _get_sent_from_preds(pred)[0]


if __name__ == '__main__':
    print(predict("D:\\py\\mp2\\mp2\\grid\\s1\\s1\\bbaf2n.mpg"))

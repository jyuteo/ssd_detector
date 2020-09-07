import numpy as np
import cv2
import tensorflow as tf


class SSDDetector():
    def __init__(self,
                 det_threshold=0.1,
                 model_path='mobilenetv2_ssd.pb',
                 gpu_memory_fraction=0.25):
        """
        Init Mobile SSD Model
        Params:
            det_threshold: prediction score threshold
            model_path: model weight path
        """
        self.det_threshold = det_threshold
        self.gpu_memory_fraction = gpu_memory_fraction
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            #Set Mobile SSD model use only 25% of GPU memory
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
            self.sess = tf.Session(graph=self.detection_graph, config=config)

    def set_threshold(self, threshold):
        """
        Change the prediction threshold score
        """
        self.det_threshold = threshold


    def predict(self, image):
        """
        Predict the face bounding boxes for given image
        Params:
            image: OpenCV image
        Returns:
            predictions: The list of face bounding boxes with confidence sorted
            scores
                [([x1,y1,x2,y2],score),(...)]
        """
        try:
            h, w, c = image.shape

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.detection_graph.get_tensor_by_name(
                'image_tensor:0')

            boxes = self.detection_graph.get_tensor_by_name(
                'detection_boxes:0')

            scores = self.detection_graph.get_tensor_by_name(
                'detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detection = self.detection_graph.get_tensor_by_name(
                'num_detections:0')

            (boxes, scores, classes, num_detection) = self.sess.run([boxes,
                                                                     scores,
                                                                     classes,
                                                                     num_detection],
                                                                    feed_dict={image_tensor: image_np_expanded})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)

            filtered_score_index = np.argwhere(
                scores >= self.det_threshold).flatten()
            selected_boxes = boxes[filtered_score_index]
            filtered_score = scores[filtered_score_index]

            faces = [[
                int(x1*w),
                int(y1*h),
                int(x2*w),
                int(y2*h),
            ] for y1, x1, y2, x2 in selected_boxes]

            predictons = []
            for (face, score) in zip(faces, filtered_score):
                predictons.append((face, score))
            # print(predictons)
            return predictons
        except Exception as e:
            print(e)
            return []

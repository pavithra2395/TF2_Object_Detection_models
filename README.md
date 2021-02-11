from google.colab import drive
drive.mount('/content/drive/')


%tensorflow_version 2.X


%cd /content/drive/MyDrive/tf2/models/research
!python setup.py build
!python setup.py install

%cd /content/drive/MyDrive/tf2/models/research/slim
!python setup.py build
!python setup.py install

%cd /content/drive/MyDrive/tf2/models/research
!protoc --python_out=. ./object_detection/protos/anchor_generator.proto ./object_detection/protos/argmax_matcher.proto ./object_detection/protos/bipartite_matcher.proto ./object_detection/protos/box_coder.proto ./object_detection/protos/box_predictor.proto ./object_detection/protos/eval.proto ./object_detection/protos/faster_rcnn.proto ./object_detection/protos/faster_rcnn_box_coder.proto ./object_detection/protos/grid_anchor_generator.proto ./object_detection/protos/hyperparams.proto ./object_detection/protos/image_resizer.proto ./object_detection/protos/input_reader.proto ./object_detection/protos/losses.proto ./object_detection/protos/matcher.proto ./object_detection/protos/mean_stddev_box_coder.proto ./object_detection/protos/model.proto ./object_detection/protos/optimizer.proto ./object_detection/protos/pipeline.proto ./object_detection/protos/post_processing.proto ./object_detection/protos/preprocessor.proto ./object_detection/protos/region_similarity_calculator.proto ./object_detection/protos/square_box_coder.proto ./object_detection/protos/ssd.proto ./object_detection/protos/ssd_anchor_generator.proto ./object_detection/protos/string_int_label_map.proto ./object_detection/protos/train.proto ./object_detection/protos/keypoint_box_coder.proto ./object_detection/protos/multiscale_anchor_generator.proto ./object_detection/protos/graph_rewriter.proto ./object_detection/protos/calibration.proto ./object_detection/protos/flexible_grid_anchor_generator.proto


%cd /content/drive/MyDrive/tf2/models/research/object_detection
!python xml_to_csv.py

%cd /content/drive/MyDrive/tf2/models/research/object_detection
!python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
!python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record

!tensorboard --logdir=training

!python model_main_tf2.py --model_dir=training --pipeline_config_path=training/faster_rcnn_resnet

%cd /content/drive/MyDrive/tf2/models/research/object_detection
#!python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.config --trained_checkpoint_prefix training/ckpt-8 --output_directory inference_graph
!python exporter_main_v2.py --trained_checkpoint_dir=training --output_directory=inference_graph --pipeline_config_path=training/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.config

%cd /content/drive/MyDrive/tf2/models/research/object_detection
!python Object_detection_image.py

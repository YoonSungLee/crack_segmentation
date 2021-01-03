# initial inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/images -model_path models/model_unet_vgg_16_best.pt -out_viz_dir test_results/initial/visual -out_pred_dir test_results/initial/pred -model_type vgg16

# initial evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/masks -pred_dir test_results/initial/pred

# train(resnet101) --> exp01
python train_unet.py -n_epoch 100 -data_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/train -model_dir models/resnet101-5d3b4d8f -num_workers 0 -model_type resnet101

# exp01 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/images -model_path models/exp01/model_best.pt -out_viz_dir test_results/exp01/visual -out_pred_dir test_results/exp01/pred -model_type resnet101

# exp01 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/masks -pred_dir test_results/exp01/pred

# train(initial) --> exp02
python train_unet.py -n_epoch 100 -data_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/train -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16
python train_unet.py -n_epoch 100 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/train -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16

# exp02 infernece
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/images -model_path models/exp02_re/model_best.pt -out_viz_dir test_results/exp02_re/visual -out_pred_dir test_results/exp02_re/pred -model_type vgg16

# exp02 inference -- he test
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/clahe/test/images -model_path models/exp02/model_best.pt -out_viz_dir test_results/exp02_testhe/visual -out_pred_dir test_results/exp02_testhe/pred -model_type vgg16

# exp02 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/masks -pred_dir test_results/exp02_re/pred

# exp02 evaluate -- he test
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/masks -pred_dir test_results/exp02_testhe/pred

# train(exp02, brightness aug) --> exp03
python train_unet.py -n_epoch 100 -data_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/train -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -augmentation True

# exp03 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/images -model_path models/exp03/model_best.pt -out_viz_dir test_results/exp03/visual -out_pred_dir test_results/exp03/pred -model_type vgg16

# exp03 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/masks -pred_dir test_results/exp03/pred

# train(initial, cracktree200 aug) --> exp04
python train_unet.py -n_epoch 100 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/train_train -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16

# exp04 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/images -model_path models/exp04/model_best.pt -out_viz_dir test_results/exp04/visual -out_pred_dir test_results/exp04/pred -model_type vgg16

# exp04 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/masks -pred_dir test_results/exp04/pred

# train(initial, clahe) --> exp05
python train_unet.py -n_epoch 100 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/clahe/train_train -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16

# exp05 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/images -model_path models/exp05/model_best.pt -out_viz_dir test_results/exp05/visual -out_pred_dir test_results/exp05/pred -model_type vgg16

# exp05 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/masks -pred_dir test_results/exp05/pred

# exp05 inference_02
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/clahe/test/images -model_path models/exp05/model_best.pt -out_viz_dir test_results/exp05_02/visual -out_pred_dir test_results/exp05_02/pred -model_type vgg16

# exp05 evaluate_02
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/masks -pred_dir test_results/exp05_02/pred

# train(initial, cracktree200) --> exp06
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16

# exp06 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/images -model_path models/exp06/model_epoch_25.pt -out_viz_dir test_results/exp06/visual -out_pred_dir test_results/exp06/pred -model_type vgg16

# exp06 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/masks -pred_dir test_results/exp06/pred

# train(initial, cracktree200, aug) --> exp07
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16

# exp07 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/images -model_path models/exp07/model_best.pt -out_viz_dir test_results/exp07/visual -out_pred_dir test_results/exp07/pred -model_type vgg16

# exp07 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/masks -pred_dir test_results/exp07/pred

# train(initial, cracktree200, aug, focal) --> exp08
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft focal

# exp08 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/images -model_path models/exp08/model_best.pt -out_viz_dir test_results/exp08/visual -out_pred_dir test_results/exp08/pred -model_type vgg16

# exp08 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/masks -pred_dir test_results/exp08/pred

# train(initial, cracktree200, aug, infogain) --> exp09
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft infogain

# exp09 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/images -model_path models/exp09/model_best.pt -out_viz_dir test_results/exp09/visual -out_pred_dir test_results/exp09/pred -model_type vgg16

# exp09 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/masks -pred_dir test_results/exp09/pred

# train(initial, cracktree200, aug, dice) --> exp10
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft dice

# exp10 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/images -model_path models/exp10/model_best.pt -out_viz_dir test_results/exp10/visual -out_pred_dir test_results/exp10/pred -model_type vgg16

# exp10 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/masks -pred_dir test_results/exp10/pred

# train(initial, cracktree200, aug, logcoshdice) --> exp11
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp11 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/images -model_path models/exp11/model_best.pt -out_viz_dir test_results/exp11/visual -out_pred_dir test_results/exp11/pred -model_type vgg16

# exp11 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/cracktree200/test/masks -pred_dir test_results/exp11/pred

# train(initial, cracktree200, aug,focaltversky) --> exp12
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft focaltversky

# exp12 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/images -model_path models/exp12/model_best.pt -out_viz_dir test_results/exp12/visual -out_pred_dir test_results/exp12/pred -model_type vgg16

# exp12 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/masks -pred_dir test_results/exp12/pred

# train(initial, minimal, aug, infogain) --> exp13
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft infogain

# exp13 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/images -model_path models/exp13/model_best.pt -out_viz_dir test_results/exp13/visual -out_pred_dir test_results/exp13/pred -model_type vgg16

# exp13 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/masks -pred_dir test_results/exp13/pred

# exp13 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp13/model_best.pt -out_viz_dir predict_results/exp13/visual -out_pred_dir predict_results/exp13/pred -model_type vgg16

# train(initial, minimal, aug, dice) --> exp14
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft dice

# exp14 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/images -model_path models/exp14/model_best.pt -out_viz_dir test_results/exp14/visual -out_pred_dir test_results/exp14/pred -model_type vgg16

# exp14 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/masks -pred_dir test_results/exp14/pred

# train(initial, minimal, aug, loogcoshdice) --> exp15
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp15 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/images -model_path models/exp15/model_best.pt -out_viz_dir test_results/exp15/visual -out_pred_dir test_results/exp15/pred -model_type vgg16

# exp15 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/masks -pred_dir test_results/exp15/pred

# train(initial, minimal, aug, focaltversky) --> exp16
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/val -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft bcedice

# exp16 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/images -model_path models/exp16/model_best.pt -out_viz_dir test_results/exp16/visual -out_pred_dir test_results/exp16/pred -model_type vgg16

# exp16 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal/test/masks -pred_dir test_results/exp16/pred

# train(initial, minimal_02, aug, bcedice) --> exp17
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft bcedice

# exp17 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/images -model_path models/exp17/model_best.pt -out_viz_dir test_results/exp17/visual -out_pred_dir test_results/exp17/pred -model_type vgg16

# exp17 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/masks -pred_dir test_results/exp17/pred

# exp17 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp17/model_best.pt -out_viz_dir predict_results/exp17/visual -out_pred_dir predict_results/exp17/pred -model_type vgg16

# train(initial, minimal_02, aug, focaltversky) --> exp18
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft focaltversky

# exp18 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/images -model_path models/exp18/model_best.pt -out_viz_dir test_results/exp18/visual -out_pred_dir test_results/exp18/pred -model_type vgg16

# exp18 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/masks -pred_dir test_results/exp18/pred

# exp18 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp18/model_best.pt -out_viz_dir predict_results/exp18/visual -out_pred_dir predict_results/exp18/pred -model_type vgg16

# train(initial, minimal_02, aug, infogain) --> exp19
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft infogain

# exp19 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/images -model_path models/exp19/model_best.pt -out_viz_dir test_results/exp19/visual -out_pred_dir test_results/exp19/pred -model_type vgg16

# exp19 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/masks -pred_dir test_results/exp19/pred

# exp19 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp19/model_best.pt -out_viz_dir predict_results/exp19/visual -out_pred_dir predict_results/exp19/pred -model_type vgg16

# train(initial, minimal_02, aug, logcoshdice) --> exp20
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp20 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/images -model_path models/exp20/model_best.pt -out_viz_dir test_results/exp20/visual -out_pred_dir test_results/exp20/pred -model_type vgg16
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp20/model_best.pt -out_viz_dir test_results/exp20_add/visual -out_pred_dir test_results/exp20_add/pred -model_type vgg16


# exp20 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/masks -pred_dir test_results/exp20/pred
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/masks -pred_dir test_results/exp20_add/pred

# exp20 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/overlap_sky -model_path models/exp20/model_best.pt -out_viz_dir predict_results/exp20_2/visual -out_pred_dir predict_results/exp20_2/pred -model_type vgg16

# train(initial, minimal_02, aug, dice) --> exp21
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft dice

# exp21 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/images -model_path models/exp21/model_best.pt -out_viz_dir test_results/exp21/visual -out_pred_dir test_results/exp21/pred -model_type vgg16

# exp21 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_02/test/masks -pred_dir test_results/exp21/pred

# exp21 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp21/model_best.pt -out_viz_dir predict_results/exp21/visual -out_pred_dir predict_results/exp21/pred -model_type vgg16

# train(initial, minimal_02, online, logcoshdice) --> exp22
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_03/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_03/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp22 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_03/test/images -model_path models/exp22/model_best.pt -out_viz_dir test_results/exp22/visual -out_pred_dir test_results/exp22/pred -model_type vgg16

# exp22 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_03/test/masks -pred_dir test_results/exp22/pred

# exp22 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp22/model_best.pt -out_viz_dir predict_results/exp22/visual -out_pred_dir predict_results/exp22/pred -model_type vgg16

# train(initial, minimal_02, online, logcoshdice) --> exp23
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_04/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_03/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp23 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_04/test/images -model_path models/exp23/model_best.pt -out_viz_dir test_results/exp23/visual -out_pred_dir test_results/exp23/pred -model_type vgg16

# exp23 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_04/test/masks -pred_dir test_results/exp23/pred

# exp23 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp23/model_best.pt -out_viz_dir predict_results/exp23/visual -out_pred_dir predict_results/exp23/pred -model_type vgg16

# train(initial, minimal_05, logcoshdice) --> exp24
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_05/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_05/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp24 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_05/test/images -model_path models/exp24/model_best.pt -out_viz_dir test_results/exp24/visual -out_pred_dir test_results/exp24/pred -model_type vgg16
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp24/model_best.pt -out_viz_dir test_results/exp24_add/visual -out_pred_dir test_results/exp24_add/pred -model_type vgg16

# exp24 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_05/test/masks -pred_dir test_results/exp24/pred
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/masks -pred_dir test_results/exp24_add/pred

# ex24 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp24/model_best.pt -out_viz_dir predict_results/exp24/visual -out_pred_dir predict_results/exp24/pred -model_type vgg16

# train(initial, minimal_06, logcoshdice) --> exp25
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp25 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp25/model_best.pt -out_viz_dir test_results/test/visual -out_pred_dir test_results/test/pred -model_type vgg16

# exp25 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/masks -pred_dir test_results/exp25/pred

# exp25 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp25/model_best.pt -out_viz_dir predict_results/exp25/visual -out_pred_dir predict_results/exp25/pred -model_type vgg16

# train(initial, minimal_06, grayscale, logcoshdice) --> exp26
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice -augmentation True

# exp26 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp26/model_best.pt -out_viz_dir test_results/exp26/visual -out_pred_dir test_results/exp26/pred -model_type vgg16


# exp26 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/masks -pred_dir test_results/exp26/pred

# exp26 predict
python predict.py -img_dir /home/jovyan/work/hangman/dataset/crack_from_project -model_path models/exp26/model_best.pt -out_viz_dir predict_results/exp26/visual -out_pred_dir predict_results/exp26/pred -model_type vgg16
python predict.py -img_dir /home/jovyan/work/hangman/dataset/overlap_sky -model_path models/exp26/model_best.pt -out_viz_dir predict_results/exp26_sky/visual -out_pred_dir predict_results/exp26_sky/pred -model_type vgg16

# train(vgg16_bn, minimal_06, logcoshdic) --> exp27
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16_bn -lossft logcoshdice

# exp27 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp27/model_best.pt -out_viz_dir test_results/exp27/visual -out_pred_dir test_results/exp27/pred -model_type vgg16_bn

# exp27 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/masks -pred_dir test_results/exp27/pred

# train(vgg16_bn_do, minimal_06, logcoshdice) --> exp28
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16_bn_do -lossft logcoshdice

# exp28 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp28/model_best.pt -out_viz_dir test_results/exp28/visual -out_pred_dir test_results/exp28/pred -model_type vgg16_bn_do

# exp28 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/masks -pred_dir test_results/exp28/pred

# train(vgg16_bn_do, minimal_06, logcoshdice) --> exp29
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16_bn_do -lossft logcoshdice

# exp29 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp29/model_best.pt -out_viz_dir test_results/exp29/visual -out_pred_dir test_results/exp29/pred -model_type vgg16_bn_do

# exp29 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/masks -pred_dir test_results/exp29/pred

# train(vgg16_fullbn_do, minimal_06, logcoshdice) --> exp30
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16_fullbn_do -lossft logcoshdice

# exp30 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp30/model_best.pt -out_viz_dir test_results/exp30/visual -out_pred_dir test_results/exp30/pred -model_type vgg16_fullbn_do

# exp30 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/masks -pred_dir test_results/exp30/pred

# train(unet+++, minimal_06, logcoshdice) --> exp31
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/valid -model_dir models/exp31 -num_workers 0 -model_type unet+++ -lossft logcoshdice -batch_size 2

# exp31 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp31/model_best.pt -out_viz_dir test_results/exp31/visual -out_pred_dir test_results/exp31/pred -model_type unet+++

# exp31 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/masks -pred_dir test_results/exp31/pred

# train(vgg16_bn_do, minimal_tmp(without cracktree200), logcoshdice) --> exp32
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_07/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_07/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp32 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_07/test/images -model_path models/exp32/model_best.pt -out_viz_dir test_results/exp32/visual -out_pred_dir test_results/exp32/pred -model_type vgg16

# exp32 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_07/test/masks -pred_dir test_results/exp32/pred

# train(vgg16_bn_do, minimal_07, logcoshdice) --> exp33
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_07/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_07/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp33 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_07/test/images -model_path models/exp33/model_best.pt -out_viz_dir test_results/exp33/visual -out_pred_dir test_results/exp33/pred -model_type vgg16

# exp33 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_07/test/masks -pred_dir test_results/exp33/pred

# train(vgg16, minimal_08, logcoshdice) --> exp34
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp34 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/test/images -model_path models/exp34/model_best.pt -out_viz_dir test_results/exp34/visual -out_pred_dir test_results/exp34/pred -model_type vgg16
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp34/model_best.pt -out_viz_dir test_results/exp34_2/visual -out_pred_dir test_results/exp34_2/pred -model_type vgg16

# exp34 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/test/masks -pred_dir test_results/exp34/pred

# train(vgg16, minimal_08, logcoshdice) --> exp35
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp35 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/test/images -model_path models/exp35/model_best.pt -out_viz_dir test_results/exp35/visual -out_pred_dir test_results/exp35/pred -model_type vgg16
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp35/model_best.pt -out_viz_dir test_results/exp35_2/visual -out_pred_dir test_results/exp35_2/pred -model_type vgg16

# exp35 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/test/masks -pred_dir test_results/exp35/pred

# train(vgg16, minimal_08, logcoshdice) --> exp36
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/valid -model_dir models/model_unet_vgg_16_best -num_workers 0 -model_type vgg16 -lossft logcoshdice

# exp36 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/test/images -model_path models/exp36/model_best.pt -out_viz_dir test_results/exp36/visual -out_pred_dir test_results/exp36/pred -model_type vgg16
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp36/model_best.pt -out_viz_dir test_results/exp36_2/visual -out_pred_dir test_results/exp36_2/pred -model_type vgg16

# exp36 evaluate
python evaluate_unet.py -ground_truth_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_08/test/masks -pred_dir test_results/exp36_2/pred

# train(vgg16, resize, minimal_06, logcoshdice) --> exp37
python train_unet.py -n_epoch 500 -train_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/train -val_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/valid -model_dir models/exp37 -num_workers 0 -model_type vgg16 -lossft logcoshdice -augmentation True -batch_size 3

# exp37 inference
python inference_with_true.py  -img_dir /home/jovyan/work/hangman/dataset/crack_segmentation_dataset/minimal_06/test/images -model_path models/exp37/model_best.pt -out_viz_dir test_results/exp37/visual -out_pred_dir test_results/exp37/pred -model_type vgg16


### ES patience 설정 확인
### width, height 설정 확인
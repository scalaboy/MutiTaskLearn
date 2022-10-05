# Statement
 this is the official code for sigir 2022 paper 《CTnoCVR: A Novelty Auxiliary Task Making the Lower-CTR-Higher-CVR Upper》

 paper is here https://dl.acm.org/doi/pdf/10.1145/3477495.3531843


# structure of the code
  * the main code is in `mctr`
  * scripts for running is in `tools`
  * put as much as possible parameters in a yaml file in `configs`, which makes it easier to manage and also change in commandline.


# how to run

* all parameters should be in a yaml file as shown in `configs`;  you have to change location of training/eval files
* `tools/train.py` control the workflow.
* python train.py -c configs/esmm_ali_cpp.yml
* you can override parameters as follows:
   * python esmm_model.py -c configs/esmm_ali_cpp.yml -o Global.epochs=4 Models.esmm.mlp_ctr="[360, 200, 80]"
   * more examples
      - CUDA_VISIBLE_DEVICES=2 python tools/train.py -c configs/esmm_ali_cpp.yml -o Loss.weights="[1., 1., 1.]" Loss.value="[direct_auc_loss, direct_auc_loss, fake_loss]"

# TODO
* figure out how to utilize the value of each feature (currently didn't pay attention to it)
* make datareader faster (currently 350ms/batch(5000 example))
* all negative batch could impact the learning process since it's easy to overfit due to the imbalance of ali-cpp dataset. I guess we could try to give a dynamic weight of each batch according to the pecentage of negative examples in a batch.  added focalLoss, but it's only comparable with binary_crossentropy loss.

# contributor
 jeffzhengye
 scalaboy

import argparse
from data_preprocess.data_builder import SummaryDataModule
from models.bart import BartOrigin
from models.t5 import T5Origin, T5MultiModal
from models.multi_modal_model import BartMultiModal
from models.tri_modal_model import BartTriModal     # 제안 모델 import
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin

 
if __name__ == '__main__':
    # for training
    parser = argparse.ArgumentParser()
    # tri_modal_bart 새로 추가
    parser.add_argument('-model', default='tri_modal_t5', type=str, help='We have for models to choose, text_only_bart, multi_modal_bart,  text_only_t5 and multi_modal_t5, tri_modal_bart')
    parser.add_argument('-checkpoint', default ='/root/VG-GPLMs/src/checkpoint/_30_tri_modal_bart/last.ckpt', type=str, help='The checkpoint path') # 처음 시작할때는 가져올 checkpoint 없음. default = '' 이거 하면 시작도 안됨. '-checkpoint', default='여기다가 ㄱㄱ', type=str,
    parser.add_argument('-train_src_path', default='/root/VG-GPLMs/src/dataset30/sum_train_30/tran.tok.txt', type=str, help='training input data path (dialogue)') #####여기서부터 train 시에 바꿀필요 없음
    parser.add_argument('-train_tgt_path', default='/root/VG-GPLMs/src/dataset30/sum_train_30/desc.tok.txt', type=str, help='training output data path (summary)') 
    parser.add_argument('-val_src_path', default='/root/VG-GPLMs/src/dataset30/sum_valid_30/tran.tok.txt', type=str, help='validatioin input data path (dialogue)')
    parser.add_argument('-val_tgt_path', default='/root/VG-GPLMs/src/dataset30/sum_valid_30/desc.tok.txt', type=str, help='validatioin output data path (summary)')
    parser.add_argument('-test_src_path', default='/root/VG-GPLMs/src/dataset30/sum_test_30/tran.tok.txt', type=str, help='testing input data path (dialogue)')
    parser.add_argument('-test_tgt_path', default='/root/VG-GPLMs/src/dataset30/sum_test_30/desc.tok.txt', type=str, help='testing output data path (summary)')
    parser.add_argument('-image_feature_path', default='/root/VG-GPLMs/src/dataset30/video_features_30/', type=str, help='video features path') #####여기까지는 train 시에 바꿀필요 없음
    # 새로 추가
    parser.add_argument('-audio_feature_path', default='/root/VG-GPLMs/src/dataset30/audio_30/concat/', type=str, help='audio features path') #####여기까지는 train 시에 바꿀필요 없음(새롭게 생긴 줄)
    parser.add_argument('-val_save_file', default='/root/VG-GPLMs/src/evaluation/temp_valid_file_30_tri_modal_bart', type=str, help='the validation results for each epoch') # 기존은 /temp_valid_file
    parser.add_argument('-test_save_file', default='./evaluation/results/test_summaries.txt', type=str, help='the generated summary for testing data') # 기존은 /test_summaries.txt
    parser.add_argument('-log_name', default='tri_modal_t5', type=str, help='lightning log path')   # 모델에 따라 바꿔줘야함. lightning log로 저장해줌 
    parser.add_argument('-gpus', default='0', type=str, help='choose gpus to run the code, you can choose multipple gpus') # 0,1,2,3 했었어야함 그래야 4개
    parser.add_argument('-batch_size', type=int, default=8, help='batch size for each gpu')     # data30hours batch4로 시작해보겠음!
    parser.add_argument('-max_input_len', type=int, default=512, help='the maximun length for input dialogue')
    parser.add_argument('-max_output_len', type=int, default=64, help='the maximun length for output summary')
    parser.add_argument('-max_img_len', type=int, default=256, help='the maximun length for video features')
    # 새로 추가 (몇으로? (5250, 43) data도 있기 때문에 256은 너무 작을듯. 일단 해보겠음)
    parser.add_argument('-max_aud_len', type=int, default=10000, help='the maximun length for audio features')   # 256에서 바꿔봄 -> 3만 
    parser.add_argument('-n_beams', type=int, default=5, help='the number of beams using for generation') # 원랜4 mulmobart 5
    parser.add_argument('-no_repeat_ngram_size', type=int, default=3, help='the size of no repeat ngrams during generation')
    parser.add_argument('-learning_rate', default=3e-5, type=float, help='learning rate')       # 원래는 3e-5
    parser.add_argument('-scheduler_lambda1', default=20, type=int, help='change the learning each lambda1 epoch')
    parser.add_argument('-scheduler_lambda2', default=0.95, type=float, help='the learning rate will times lambda2 for each change')
    parser.add_argument('-num_epochs', type=int, default=300, help='maximun number of training epoches') # 원래는100
    parser.add_argument('-grad_accumulate', type=int, default=10, help='gradient accumulation for this number iterations')
    parser.add_argument('-random_seed', type=int, default=0, help='global random seed')
    parser.add_argument('-do_train', type=str, default='False', help='set True to training, set False to not training')
    parser.add_argument('-do_test', type=str, default='True', help='set True to testing, set False to not testing') # 일단 train 만이니까 test는 False 해놓겠음
    parser.add_argument('-limit_val_batches', default=1.0, type=float, help='do validation for each epoch')
    parser.add_argument('-val_check_interval', type=float, default=1, help='do validation for each epoch')
    parser.add_argument('-img_lr_factor', type=float, default=5, help='the learning rate for visual guidance part will times this number') # 원랜1 멀모바트 5
    # 여기도 새로 추가
    parser.add_argument('-aud_lr_factor', type=float, default=5, help='the learning rate for audio guidance part will times this number') # 원랜1 멀모바트 5
    
    # About cross-modal attention and fusion
    parser.add_argument('-use_img_trans', action='store_true', help='whether or not to use VTF')    # python ~~.py "-use_img_trans" 플래그 안적으면 args.use_img_trans 변수에 False가 할당. 
    parser.add_argument('-use_forget_gate', action='store_true', help='whether or not to use forget gate')  # 요 놈도 마찬가지. 쓰려면 꼭 플래그 적기. dafault='True'해놔서 걍 해도 적용
    parser.add_argument('-fusion_layer', type=int, default=5, help='number of fusion layers') # 5 is the last layer
    '''
    <참고>
    Textual features: T in (S_t, D_t)
    Visual features: V in (S_v, D_v)
    Audio features : A in (S_a, D_a)  # Dimmension = 43

    cross_attn_type == 0
    => V' = linear_1(V) in (S_v, D_t)
    => A = Dot_Prod_Attn(T, V') in (S_t, S_v)
    => A = softmax(A)
    => T_out = linear_2(concat(T, AV)) in (S_t, D_t)
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 1
    => V' = linear_1(V) in (S_v, D_t)
    => A = Dot_Prod_Attn(T, V') in (S_t, S_v)
    => A = softmax(A)
    => T_out = linear_2(concat(T, AV')) in (S_t, D_t), only this step is different from 0
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 2
    => V' = linear_1(V) in (S_v, D_t)
    => A = Dot_Prod_Attn(T, V') in (S_t, S_v)
    => A = softmax(A)
    => T_out = AV'
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 3
    => T' = linear_1(T) in (S_t, D_a), D_a << D_t
    => V' = linear_2(V) in (S_v, D_a), D_a << D_v
    => A = Dot_Prod_Attn(T', V') in (S_t, S_v)
    => A = softmax(A)
    => T_out = linear_3(concat(T, AV)) in (S_t, D_t)
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 4
    => K_1 = linear_1(V) in (S_v, common)
    => V_1 = linear_2(V) in (S_v, common)
    => Q_1 = linear_3(T) in (S_t, common)
    => T_out = MultiHeadAttn(Q_1, K_1, V_1) in (S_t, common)
    => T_out = linear_4(concat(T, T_out)) in (S_t, D_t)
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 5 (Only valid when D_a == D_t)
    => K_a = linear_1(V) in (S_v, D_a)
    => V_a = linear_2(V) in (S_v, D_a)
    => Q_a = linear_3(T) in (S_t, D_a)
    => T_out = MultiHeadAttn(Q_a, K_a, V_a) in (S_t, D_a)
    => T_out = T + T_out (Residual Connection)

    ################# 여기서부터 추가 #################
    cross_attn_type == 6 (*Proposed Trimodal model)
    => K_1 = linear_1(V) in (S_v, common1)              K = (Vision길이, batch, common)
    => V_1 = linear_2(V) in (S_v, common1)              V = (vision길이, batch, common) 
    => Q_1 = linear_3(T) in (S_t, common1)              Q = (text길이, batch, common)
    => T_out = MultiHeadAttn(Q_1, K_1, V_1) in (S_t, common1)   
    => T_out = linear_4(concat(T, T_out)) in (S_t, D_t)     
    -----------------> 1차 멀티통과후 output = (batch, S_t, D_t)     *여기서 D_t = text dimm = 768
    => K_2 = linear_5(A) in (S_a, common2)              K = (Audio길이, batch, common)  *여기서 common1,2은 각각 바꿀수 있을듯 * A : Audio
    => V_2 = linear_6(A) in (S_a, common2)  
    => Q_2 = linear_7(T) in (S_t, common2)              Q = (text길이, batch, common)
    => T_out = MultiHeadAttn(Q_2, K_2, V_2) in (S_t, common2)
    => T_out = linear_8(concat(T, T_out)) in (S_t, common2)



    <참고> 
    Textual features: T in (S_t, D_t)   # D_t = 768
    Visual features: V in (S_v, D_v)    # D_v = 2048
    Audio features : A in (S_a, D_a)    # D_a = 43
    '''
    parser.add_argument('-cross_attn_type', type=int, default=4)  # 현재 text_only_bart라서 0으로 넣어놈 # text_only_T5 0 하겠음. 
    parser.add_argument('-dim_common', type=int, default=256) ##
    parser.add_argument('-n_attn_heads', type=int, default=2) # batch32 head4  --- batch16 head8  ----- batch8 head16
                        # attention head 수를 임베딩 차원의 약수가 되어야함.
    # Add to decoding
    parser.add_argument('-fusion_in_decoding', action='store_true')
    parser.add_argument('-vision_use_noise', action='store_true')

    args = parser.parse_args()

    # random seed
    seed_everything(args.random_seed)

    # set logger
    logger = pl_loggers.TensorBoardLogger(f'./lightning_logs/_30_tri_modal_bart') #    logger = pl_loggers.TensorBoardLogger(f'./lightning_logs/{args.log_name}')

    # save checkpoint
    checkpoint_callback = ModelCheckpoint('./checkpoint/_30_tri_modal_bart',
                                          monitor='validation_Rouge2_one_epoch',
                                          save_last=True,
                                          save_top_k=2,
                                          mode='max',
                                          )

    # make trainer
    if args.checkpoint == 'None':
        args.checkpoint = None
    trainer = Trainer(deterministic=True,
                      num_sanity_val_steps=10,
                      resume_from_checkpoint=args.checkpoint,
                      logger=logger,
                      gpus=args.gpus,
                      distributed_backend='ddp',
                      plugins=DDPPlugin(find_unused_parameters=False),
                      gradient_clip_val=1.0,
                      max_epochs=args.num_epochs,
                      limit_val_batches=args.limit_val_batches,
                      val_check_interval=args.val_check_interval,
                      accumulate_grad_batches=args.grad_accumulate,
                      fast_dev_run=False,
                      callbacks=[checkpoint_callback])

    # make dataloader & model
    summary_data = SummaryDataModule(args)  # 일단 여기서 걸리게
    if args.model == 'text_only_bart':
        model = BartOrigin(args)
    elif args.model == 'multi_modal_bart':
        model = BartMultiModal(args)
    elif args.model == 'tri_modal_bart' :
        model = BartTriModal(args)
    elif args.model == 'text_only_t5':
        model = T5Origin(args)
    elif args.model == 'multi_modal_t5':
        model = T5MultiModal(args)
    else:
        raise ValueError("Invalid model")

    # Fit the instantiated model to the data
    if args.do_train == 'True':
        trainer.fit(model, train_dataloader=summary_data.train_loader, val_dataloaders=summary_data.val_loader)
    if args.do_test == 'True':
        model = model.load_from_checkpoint(args.checkpoint, args=args)
        trainer.test(model=model, test_dataloaders=summary_data.test_loader)
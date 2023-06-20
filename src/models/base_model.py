
import pytorch_lightning as pl
import torch

class BaseModel(pl.LightningModule):

    def __init__(self, args):   
        # 클래스 인스턴스를 생성할때 입력받는 args를 저장하고 있음
        super().__init__()
        self.args = args
        self.learning_rate = args.learning_rate

    def forward(self):
        pass
        #forward 메소드는 모델의 forward 연산을 수행하는 메소드로, 
        # 이 메소드에서 모델이 어떻게 동작할지 정의합니다. 
        # 해당 코드에서는 pass 문장만 포함되어 있어 forward 연산이 수행되지는 않음. 


    ### training_step, validation_step, validation_epoch_end, test_epoch_end 모두 
    ### pytorch lightning 에서 제공하는 Training Loop 단계
    def training_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
        # logs
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        # 학습 데이터(batch)를 받아서 forward 연산을 수행하고, 손실(loss)을 계산하고, 
        # 손실값을 logging합니다.
         

    def validation_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
        # 검증 데이터(batch)를 받아서 forward 연산을 수행하고, 손실을 계산합니다. 
        # 이 때 self.log 메소드를 이용하여 검증 데이터 손실을 logging하고 있습니다.

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('val_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True)
        #  epoch이 끝난 후 전체 검증 데이터의 평균 손실값을 계산하고, 
        #  self.log 메소드를 이용하여 전체 검증 데이터 손실값을 logging하고 있습니다

    def test_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
        return loss
        # 테스트 데이터(batch)를 받아서 forward 연산을 수행하고, 손실을 계산합니다.

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('test_loss', avg_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.args.img_lr_factor != 1 and self.args.model=='multi_modal_bart':
            # make parameter groups
            all_para = [p for p in self.model.parameters()]
            # img_related_para = [p for p in self.model.model.encoder.img_transformer.parameters()] \
            #                   +[p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]
            # 이미지 관련 파라미터를 설정하기 위해 _img_related_para 리스트를 만들고, 
            # cross_attn_type에 따라 적절한 파라미터들을 _img_related_para에 추가하고 있습니다.


            # img_related_para = [p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]

            _img_related_para = []
            if self.args.cross_attn_type == 0:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 1:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 2:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters()
                ]
            elif self.args.cross_attn_type == 3:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters()
                ]
            elif self.args.cross_attn_type == 4:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters(),
                    self.model.model.encoder._linear_4.parameters(),
                    self.model.model.encoder._multi_head_attn.parameters()
                ]
            elif self.args.cross_attn_type == 5:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters(),
                    self.model.model.encoder._multi_head_attn.parameters()
                ]
            

            if self.args.use_forget_gate:
                _img_related_para.append(self.model.model.encoder.fg.parameters())

            img_related_para = []
            for params in _img_related_para:
                for param in params:
                    img_related_para.append(param)

            bart_para = []
            # bart_para 리스트는 이미지 관련 파라미터를 제외한 모든 파라미터들을 담은 리스트입니다.
            # 이 리스트는 최종적으로 optimizer에 전달될 파라미터 그룹입니다. 
            # 이 파라미터 그룹은 BART 모델의 파라미터 그룹입니다. (왜냐면 if문에 multimodal_bart이거든)

            for p in all_para:
                flag = 0
                for q in img_related_para:
                    if p.shape == q.shape:
                        if torch.equal(p, q):
                            flag = 1
                if flag == 0:
                    bart_para.append(p)
                    continue

            optimizer = torch.optim.Adam([
                {'params': bart_para},
                {'params': img_related_para, 'lr': self.learning_rate * self.args.img_lr_factor},
            ], lr=self.learning_rate)
            ## torch.optim.Adam 함수를 호출하여 optimizer 객체를 생성합니다. 
            # 이 함수는 BART 모델의 파라미터 그룹과 이미지 관련 파라미터의 파라미터 그룹을 따로 설정하고 있습니다. 
            # BART 모델의 파라미터 그룹은 bart_para 리스트에서 가져오고, 
            # 이미지 관련 파라미터의 파라미터 그룹은 img_related_para 리스트에서 가져옵니다. 
            # 이미지 관련 파라미터의 학습률은 self.learning_rate * self.args.img_lr_factor로 설정
            # 이 함수는 생성된 optimizer 객체를 반환합니다.

        #############################################################################################
        ############################### 여기 추가함 tri_modal_bart ##################################
        #############################################################################################
        if self.args.img_lr_factor != 1 and self.args.model=='tri_modal_bart':
            # make parameter groups
            all_para = [p for p in self.model.parameters()]
            # img_related_para = [p for p in self.model.model.encoder.img_transformer.parameters()] \
            #                   +[p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]
            # 이미지 관련 파라미터를 설정하기 위해 _img_related_para 리스트를 만들고, 
            # cross_attn_type에 따라 적절한 파라미터들을 _img_related_para에 추가하고 있습니다.


            # img_related_para = [p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]

            _img_related_para = []
            if self.args.cross_attn_type == 0:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 1:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 2:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters()
                ]
            elif self.args.cross_attn_type == 3:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters()
                ]
            elif self.args.cross_attn_type == 4:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters(),
                    self.model.model.encoder._linear_4.parameters(),
                    self.model.model.encoder._multi_head_attn.parameters()
                ]
            elif self.args.cross_attn_type == 5:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters(),
                    self.model.model.encoder._multi_head_attn.parameters()
                ]
            elif self.args.cross_attn_type == 6 :       # 바뀐부분
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters(),
                    self.model.model.encoder._linear_4.parameters(),
                    self.model.model.encoder._multi_head_attn_1.parameters(),
                    self.model.model.encoder._linear_5.parameters(),
                    self.model.model.encoder._linear_6.parameters(),
                    self.model.model.encoder._linear_7.parameters(),
                    self.model.model.encoder._multi_head_attn_2.parameters()  # 첫번째랑 달라야하나?
                ]

            if self.args.use_forget_gate:
                _img_related_para.append(self.model.model.encoder.fg.parameters())

            img_related_para = []
            for params in _img_related_para:
                for param in params:
                    img_related_para.append(param)

            bart_para = []
            # bart_para 리스트는 이미지 관련 파라미터를 제외한 모든 파라미터들을 담은 리스트입니다.
            # 이 리스트는 최종적으로 optimizer에 전달될 파라미터 그룹입니다. 
            # 이 파라미터 그룹은 BART 모델의 파라미터 그룹입니다. (왜냐면 if문에 multimodal_bart이거든)

            for p in all_para:
                flag = 0
                for q in img_related_para:
                    if p.shape == q.shape:
                        if torch.equal(p, q):
                            flag = 1
                if flag == 0:
                    bart_para.append(p)
                    continue

            optimizer = torch.optim.Adam([
                {'params': bart_para},
                {'params': img_related_para, 'lr': self.learning_rate * self.args.img_lr_factor},
            ], lr=self.learning_rate)
            ## torch.optim.Adam 함수를 호출하여 optimizer 객체를 생성합니다. 
            # 이 함수는 BART 모델의 파라미터 그룹과 이미지 관련 파라미터의 파라미터 그룹을 따로 설정하고 있습니다. 
            # BART 모델의 파라미터 그룹은 bart_para 리스트에서 가져오고, 
            # 이미지 관련 파라미터의 파라미터 그룹은 img_related_para 리스트에서 가져옵니다. 
            # 이미지 관련 파라미터의 학습률은 self.learning_rate * self.args.img_lr_factor로 설정
            # 이 함수는 생성된 optimizer 객체를 반환합니다.    

        elif self.args.img_lr_factor != 1 and self.args.model=='multi_modal_t5': 
             # make parameter groups 이번엔 T5
            all_para = [p for p in self.model.parameters()]
            # img_related_para = [p for p in self.model.model.encoder.img_transformer.parameters()] \
            #                   +[p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]

            # img_related_para = [p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]

            _img_related_para = []
            if self.args.cross_attn_type == 0:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 1:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 2:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters()
                ]
            elif self.args.cross_attn_type == 3:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters(),
                    self.model.encoder._linear_3.parameters()
                ]
            elif self.args.cross_attn_type == 4:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters(),
                    self.model.encoder._linear_3.parameters(),
                    self.model.encoder._linear_4.parameters(),
                    self.model.encoder._multi_head_attn.parameters()
                ]
            elif self.args.cross_attn_type == 5:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters(),
                    self.model.encoder._linear_3.parameters(),
                    self.model.encoder._multi_head_attn.parameters()
                ]

            if self.args.use_forget_gate:
                _img_related_para.append(self.model.encoder.fg.parameters())

            img_related_para = []
            for params in _img_related_para:
                for param in params:
                    img_related_para.append(param)

            bart_para = []
            # 여기 써있는건 bart이지만 위에 if문에서 T5를 선언해줬기때문에 T5라고 생각
            for p in all_para:
                flag = 0
                for q in img_related_para:
                    if p.shape == q.shape:
                        if torch.equal(p, q):
                            flag = 1
                if flag == 0:
                    bart_para.append(p)
                    continue

            optimizer = torch.optim.Adam([
                {'params': bart_para},
                {'params': img_related_para, 'lr': self.learning_rate * self.args.img_lr_factor},
            ], lr=self.learning_rate)
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1, gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]
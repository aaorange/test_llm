def train(sft_config:SFTConfig):
    """
    训练主流程：
    1.初始化模型，优化器，获取总的训练数据
    2.构造模型前向传播的输入，imput_ids,labels等
    3.前向传播，获取到logits
    4.基于logits,assistant_answer_mask,labels算损失
    5.做一个学习率调度
    6.基于新的学习率，做参数更新
    """

    #初始化模型
    from transformers import AutoModelForCausalLM
    from torch.optim.adamw import AdamW
    model
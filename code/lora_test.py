'''
[
{"instruction":"用户说，我的货怎么还没到？","input":"","output":"您好，请提供订单号，我来帮你查询一下时间。"},
{"instruction":"用户说：司机绕路了怎么办？","input":"","output":"对不起，我马上来跟您核实情况，如果属实会跟您进一步反馈沟通。"}
]
'''
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq)
from datasets import load_dataset
from peft import LoraConfig,get_peft_model

model_name = "./qwen-1.8b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True) #自动分配到GPU

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="right"

#加载数据集
dataset = load_dataset("json",data_files="train.jsonl")

def format_prompt(sample):
    if sample['input'] and sample['input'].strip():
        user_content = f"{sample['instruction']}\n{sample['input']}"
    else:
        user_content = sample['instruction']

    return f"""<|im_start|>user
    {user_content}<|im_end|>
    <|im_start|>assistant
    {sample['output']}<|im_end|>"""

def tokenize_function(examples):
    texts = []
    for i in range(len(examples['instruction'])):
        # 取出第 i 个样本的各个字段
        inst = examples['instruction'][i]
        inp = examples.get('input', [''] * len(examples['instruction']))[i] or ''
        out = examples['output'][i]
        
        # 构建用户消息
        if inp.strip():
            user_content = f"{inst}\n{inp}"
        else:
            user_content = inst
        
        # Qwen chatml 格式
        prompt = f"""<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{out}<|im_end|>"""
        texts.append(prompt)
    
    # 批量分词
    model_inputs = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

#处理数据集
tokenized_dateset=dataset.map(tokenize_function,batched=True,remove_columns=dataset["train"].column_names)

#配置LoRA
lora_config = LoraConfig(
        r = 16, #秩
        lora_alpha = 32,#缩放因子
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.1,#防止过拟合
        bias="none",
        task_type="CAUSAL_LM"
        )

model = get_peft_model(model,lora_config)
model.print_trainable_parameters() #查看需要训练的参数量

training_args = TrainingArguments(
        output_dir="./qwen-lora-finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=True, #混合精度训练，省内存
        save_steps=500,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False
        )

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dateset["train"],
        )

#开始训练
trainer.train()

model.save_pretrained("./qwen-lora-adapter")#保存不是完整大模型
tokenizer.save_pretrained("./qwen-lora-adapter")
print("LoRA微调完成")

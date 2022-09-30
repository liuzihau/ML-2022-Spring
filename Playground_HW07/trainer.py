import torch
import transformers
from transformers import AdamW
from tqdm.auto import tqdm

from utils import evaluate

def train_model(model, opt, tokenizer, train_loader, dev_questions, dev_loader, cv_number=None):
    optimizer = AdamW(model.parameters(), lr=float(opt.learning_rate))
    step_per_epoch = len(train_loader) // opt.accum_iter + 1
    total_step = step_per_epoch * opt.num_epoch
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,int(total_step*min(0.99,opt.warm_up_ratio)),total_step,-1)

    if opt.fp16_training:
        model, optimizer, train_loader = opt.accelerator.prepare(model, optimizer, train_loader) 

    model.train()
    best_dev_acc_avg = opt.save_baseline 
    print("Start Training ...")

    for epoch in range(opt.num_epoch):
        step = 1
        train_loss = train_acc = 0
        
        for data in tqdm(train_loader):	
            # Load all data into GPU
            data = [i.to(opt.device) for i in data]
        
            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
        
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss
        
            if opt.fp16_training:
                opt.accelerator.backward(output.loss)
            else:
                output.loss.backward()
            
            if ((step + 1) % opt.accum_iter == 0) or (step + 1 == len(train_loader)):
                optimizer.step()
                ##### TODO: Apply linear learning rate decay #####
                # optimizer.param_groups[0]["lr"] -= float(opt.learning_rate) / step_per_epoch
                # if optimizer.param_groups[0]["lr"]<-0.001:
                #    print(f"[warn]Learning rate is {optimizer.param_groups[0]['lr']}<0, please check decay method")
                scheduler.step()
                optimizer.zero_grad()
            
            step += 1
            
            # Print training loss and accuracy over past logging step
            if step % opt.logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / opt.logging_step:.3f}, acc = {train_acc / opt.logging_step:.3f}")
                train_loss = train_acc = 0
                if opt.validation:
                    print("Evaluating Dev Set ...")
                    model.eval()
                    with torch.no_grad():
                        dev_acc = 0
                        for i, data in enumerate(tqdm(dev_loader)):
                            output = model(input_ids=data[0].squeeze(dim=0).to(opt.device), token_type_ids=data[1].squeeze(dim=0).to(opt.device),
                                            attention_mask=data[2].squeeze(dim=0).to(opt.device))
                            # prediction is correct only if answer text exactly matches
                            dev_acc += evaluate(data, tokenizer, output) == dev_questions[i]["answer_text"]
                            dev_acc_avg = dev_acc / len(dev_loader)
                        print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc_avg:.3f}")
                    model.train()
                    if dev_acc_avg > best_dev_acc_avg:
                        print("Saving Best Model ...")
                        model.save_pretrained(opt.model_save_dir)
                        torch.save(model.state_dict(), f"{opt.model_save_dir}/checkpoint_{cv_number}.pt")
                        best_dev_acc_avg = dev_acc_avg
    
        if opt.validation:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    output = model(input_ids=data[0].squeeze(dim=0).to(opt.device), token_type_ids=data[1].squeeze(dim=0).to(opt.device),
                                            attention_mask=data[2].squeeze(dim=0).to(opt.device))
                    # prediction is correct only if answer text exactly matches
                    dev_acc += evaluate(data, tokenizer, output) == dev_questions[i]["answer_text"]
                print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
            model.train()

            # Save a model and its configuration file to the directory 「saved_model」 
            # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
            # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
            if dev_acc_avg > best_dev_acc_avg:
                print("Saving Best Model ...")
                model.save_pretrained(opt.model_save_dir)
                torch.save(model.state_dict(), f"{opt.model_save_dir}/checkpoint_{cv_number}.pt")
                best_dev_acc_avg = dev_acc_avg


def test_model(model, opt, tokenizer, test_questions, test_loader):
    print("Evaluating Test Set ...")
    result = []
    
    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(opt.device), token_type_ids=data[1].squeeze(dim=0).to(opt.device),
                        attention_mask=data[2].squeeze(dim=0).to(opt.device))
            result.append(evaluate(data, tokenizer, output))

    result_file = "result.csv"
    with open(result_file, 'w') as f:	
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    print(f"Completed! Result is in {result_file}")

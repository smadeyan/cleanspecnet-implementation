wandb.login(key="8e00abb67ba14a9fe0b7208b48a9200ac1fcf91c")

# Create your wandb run
run = wandb.init(
    name    = "batch-32-graph-run", ### Wandb creates random run names if you skip this field, we recommend you give useful names
    reinit  = True, ### Allows reinitalizing runs when you re-run this cell
    project = "idl-project", ### Project should be created in your wandb account
    config  = config ### Wandb Config for your run
)

# Iterate over number of epochs to train and evaluate your model
torch.cuda.empty_cache()
gc.collect()
wandb.watch(model, log="all")

best_loss = chkpt_loss

for epoch in range(50):

    print("\nEpoch {}/{}".format(epoch+1, config['epochs']))

    curr_lr                 = float(optimizer.param_groups[0]['lr'])
    train_loss   = train(model, trainloader, optimizer, criterion)
    # scheduler.step()

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))

    ### Log metrics at each epoch in your run
    # Optionally, you can log at each batch inside train/eval functions
    # (explore wandb documentation/wandb recitation)
    wandb.log({'loss': train_loss, 'lr': curr_lr})

    print({'train_loss': train_loss, 'lr': curr_lr, 'iterations': ((epoch + 1) * config['batch_size'])})

    #### For generating STOI and PESQ graphs
    if epoch % 2 == 0:
      best_loss = train_loss
      torch.save({
                'iterations'              : epoch,
                'model_state_dict'        : model.state_dict(),
                'optimizer_state_dict'    : optimizer.state_dict(),
                'loss'                    : train_loss,
                # 'scheduler_state_dict'    : scheduler.state_dict()
      }, 'specnet_checkpoints/checkpoint_{}_batch32_graph.pth'.format(epoch))

### Finish your wandb run
#run.finish()
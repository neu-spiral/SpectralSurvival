import torchtuples as tt
from methods.utils import *
from methods.metrics import *
import methods.models as models
from lassonet import LassoNetCoxRegressor
from lassonet.cox import CoxPHLoss
from pycox.models import CoxPH, CoxTime, MTLR, CoxCC, DeepHitSingle
from pycox.models.cox_time import MLPVanillaCoxTime
from functools import partial
import wandb
from methods.medicalnet import generate_model


def run_fastcph(ResNet, x_train, x_val,  x_test, y_train, y_val, y_test, learning_rate, dims, \
                                                        depth, epochs, dropout, batch_size):
    y_train, y_test, y_val = np.array(y_train).T, np.array(y_test).T, np.array(y_val).T
    model = LassoNetCoxRegressor(
        optim=(partial(torch.optim.Adam, lr=learning_rate),
                    partial(torch.optim.SGD, lr=1e-3, momentum=0.9),),
        tie_approximation="breslow",
        hidden_dims=dims,
        path_multiplier=1.01,
        torch_seed=1,
        verbose=False,
        backtrack=True)
    hist = model.path(x_train, y_train, X_val=x_val, y_val=y_val, return_state_dicts=True)

    n_records = len(hist)
    val_objs = np.zeros((n_records))
    for i in range(n_records):
        val_objs[i] = hist[i].val_objective

    model.load(hist[np.argmin(val_objs)].state_dict)

    test_AUC, test_IBS, test_CI, test_surv_MSE = sksurv_metrics(model, x_train, y_train.T, x_test, y_test.T, predict_function=True, use_exp=True, plot=False, model_name="FastCPH")
    val_AUC, val_IBS, val_CI, val_surv_MSE = sksurv_metrics(model, x_train, y_train.T, x_val, y_val.T, predict_function=True, use_exp=True)
    return test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE, hist


def run_deepsurv(ResNet, x_train, x_val,  x_test, y_train, y_val, y_test, learning_rate, dims, \
                                                        depth, epochs, dropout, batch_size, batch_norm=False, out_features=1, output_bias=False):
    val = x_val, y_val
    
    if len(x_train.shape) > 3:
        if ResNet:
            input_W = x_train.shape[-2]
            input_H = x_train.shape[-1]
            input_D = x_train.shape[2]
            net = generate_model(model='resnet', model_depth=depth, input_W=input_W, input_H=input_H, input_D=input_D, \
                                        resnet_shortcut='B', pretrain_path=f'./models/resnet_{depth}.pth')
        else:
            net = models.Conv3D_torch(out_features, depth=depth)
    else:
        in_features = x_train.shape[1]
        if ResNet:
            net = models.ResNet1D_torch()
        else:
            net = tt.practical.MLPVanilla(in_features, dims, out_features, batch_norm,
                              dropout, output_bias=output_bias)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(learning_rate)
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False
    for epoch in range(epochs):
        log = model.fit(x_train, y_train, batch_size, 1, callbacks, verbose, val_data=val, val_batch_size=batch_size)
        train_loss = log.to_pandas()["train_loss"].iloc[-1]
        val_loss = log.to_pandas()["val_loss"].iloc[-1]
        if math.isnan(train_loss) or math.isnan(val_loss):
            print(f"Epoch {epoch + 1}: Loss became NaN. Stopping training.")
            break
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    test_AUC, test_IBS, test_CI, test_surv_MSE = sksurv_metrics(model, x_train, y_train, x_test, y_test, predict_function=True, use_exp=True, plot=False, model_name="DeepSurv")
    val_AUC, val_IBS, val_CI, val_surv_MSE = sksurv_metrics(model, x_train, y_train, x_val, y_val, predict_function=True, use_exp=True)
    return test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE, log


def run_coxcc(ResNet, x_train, x_val,  x_test, y_train, y_val, y_test, learning_rate, dims, \
                                                        depth, epochs, dropout, batch_size, batch_norm=False, out_features=1, output_bias=False):
    
    val = tt.tuplefy(x_val, y_val)

    if len(x_train.shape) > 3:
        if ResNet:
            input_W = x_train.shape[-2]
            input_H = x_train.shape[-1]
            input_D = x_train.shape[2]
            model_depth = 50
            net = generate_model(model='resnet', model_depth=model_depth, input_W=input_W, input_H=input_H, input_D=input_D, \
                                        resnet_shortcut='B', pretrain_path=f'./models/resnet_{model_depth}.pth')
        else:
            net = models.Conv3D_torch(out_features, depth=depth)
    else:
        in_features = x_train.shape[1]
        if ResNet:
            net = models.ResNet1D_torch()
        else:
            net = tt.practical.MLPVanilla(in_features, dims, out_features, batch_norm,
                              dropout, output_bias=output_bias)
    model = CoxCC(net, tt.optim.Adam)
    model.optimizer.set_lr(learning_rate)
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val.repeat(10).cat())
    
    _ = model.compute_baseline_hazards()
    surv_test = model.predict_surv_df(x_test)
    surv_val = model.predict_surv_df(x_val)

    test_AUC, test_IBS, test_CI, test_surv_MSE = discrete_metrics(surv_test, y_train, y_test, plot=False, model_name="CoxCC")
    val_AUC, val_IBS, val_CI, val_surv_MSE = discrete_metrics(surv_val, y_train, y_val)
    return test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE, log


def run_coxtime(ResNet, x_train, x_val,  x_test, y_train, y_val, y_test, learning_rate, dims, \
                                                        depth, epochs, dropout, batch_size, batch_norm=False, out_features=1, output_bias=False):
    
    labtrans = CoxTime.label_transform()
    Y_train, E_train = y_train
    Y_val, E_val = y_val
    Y_test, E_test = y_test
    if len(x_train.shape) > 3:
        x_train, x_val, x_test = x_train.numpy(), x_val.numpy(), x_test.numpy()
        # Convert PyTorch tensors to NumPy arrays
        Y_train, E_train = Y_train.numpy(), E_train.numpy()
        y_train = tuple([Y_train, E_train])
        Y_val, E_val = Y_val.numpy(), E_val.numpy()
        y_val = tuple([Y_val, E_val])
        Y_test, E_test = Y_test.numpy(), E_test.numpy()
        y_test = tuple([Y_test, E_test])
    y_train_t = labtrans.fit_transform(Y_train, E_train)
    y_val_t = labtrans.transform(Y_val, E_val)

    val = tt.tuplefy(x_val, y_val_t)
    if len(x_train.shape) > 3:
        if ResNet:
            input_W = x_train.shape[-2]
            input_H = x_train.shape[-1]
            input_D = x_train.shape[2]
            model_depth = 50
            net = generate_model(model='resnet', model_depth=model_depth, input_W=input_W, input_H=input_H, input_D=input_D, \
                                        resnet_shortcut='B', pretrain_path=f'./models/resnet_{model_depth}.pth')
        else:
            net = models.Conv3dCoxTime()
    else:
        in_features = x_train.shape[1]
        if ResNet:
            net = models.ResNet1dCoxTime()
        else:
            net = MLPVanillaCoxTime(in_features, dims, batch_norm, dropout)
    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
    model.optimizer.set_lr(learning_rate)
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False
    log = model.fit(x_train, y_train_t, batch_size, epochs, callbacks, verbose,
                val_data=val.repeat(10).cat())
    
    _ = model.compute_baseline_hazards()
    surv_test = model.predict_surv_df(x_test) # shape n_train*n_test
    surv_val = model.predict_surv_df(x_val) # shape n_train*n_test

    test_AUC, test_IBS, test_CI, test_surv_MSE = discrete_metrics(surv_test, y_train, y_test, plot=False, model_name="CoxTime")
    val_AUC, val_IBS, val_CI, val_surv_MSE = discrete_metrics(surv_val, y_train, y_val)
    return test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE, log

def run_deephit(ResNet, x_train, x_val,  x_test, y_train, y_val, y_test, learning_rate, dims, \
                                                        depth, epochs, dropout, batch_size, batch_norm=False, out_features=1, output_bias=False):
    
    num_durations = 10
    labtrans = DeepHitSingle.label_transform(num_durations)

    Y_train, E_train = y_train
    Y_val, E_val = y_val
    Y_test, E_test = y_test
    if len(x_train.shape) > 3:
        x_train, x_val, x_test = x_train.numpy(), x_val.numpy(), x_test.numpy()
        # Convert PyTorch tensors to NumPy arrays
        Y_train, E_train = Y_train.numpy(), E_train.numpy()
        y_train = tuple([Y_train, E_train])
        Y_val, E_val = Y_val.numpy(), E_val.numpy()
        y_val = tuple([Y_val, E_val])
        Y_test, E_test = Y_test.numpy(), E_test.numpy()
        y_test = tuple([Y_test, E_test])
    y_train_t = labtrans.fit_transform(Y_train, E_train)
    y_val_t = labtrans.transform(Y_val, E_val)

    val = (x_val, y_val_t)
    out_features = labtrans.out_features
    if len(x_train.shape) > 3:
        if ResNet:
            input_W = x_train.shape[-2]
            input_H = x_train.shape[-1]
            input_D = x_train.shape[2]
            model_depth = 50
            net = generate_model(model='resnet', model_depth=model_depth, input_W=input_W, input_H=input_H, input_D=input_D, \
                                        resnet_shortcut='B', pretrain_path=f'./models/resnet_{model_depth}.pth')
        else:
            net = models.Conv3D_torch(out_features, depth=depth)
    else:
        in_features = x_train.shape[1]
        if ResNet:
            net = models.ResNet1D_torch(num_classes=num_durations)
        else:
            net = tt.practical.MLPVanilla(in_features, dims, out_features, batch_norm,
                              dropout, output_bias=output_bias)
    
    model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
    model.optimizer.set_lr(learning_rate)
    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(x_train, y_train_t, batch_size, epochs, callbacks, verbose=False, val_data=val)

    surv_test = model.predict_surv_df(x_test)
    surv_val = model.predict_surv_df(x_val)

    test_AUC, test_IBS, test_CI, test_surv_MSE = discrete_metrics(surv_test, y_train, y_test, plot=False, model_name="DeepHit")
    val_AUC, val_IBS, val_CI, val_surv_MSE = discrete_metrics(surv_val, y_train, y_val)
    
    return test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE, log


def run_MTLR(ResNet, x_train, x_val, x_test, y_train, y_val, y_test, learning_rate, dims, \
                                                        depth, epochs, dropout, batch_size, batch_norm=False, out_features=1, output_bias=False):
    num_durations = 10
    labtrans = MTLR.label_transform(num_durations)
    Y_train, E_train = y_train
    Y_val, E_val = y_val
    Y_test, E_test = y_test
    if len(x_train.shape) > 3:
        x_train, x_val, x_test = x_train.numpy(), x_val.numpy(), x_test.numpy()
        # Convert PyTorch tensors to NumPy arrays
        Y_train, E_train = Y_train.numpy(), E_train.numpy()
        y_train = tuple([Y_train, E_train])
        Y_val, E_val = Y_val.numpy(), E_val.numpy()
        y_val = tuple([Y_val, E_val])
        Y_test, E_test = Y_test.numpy(), E_test.numpy()
        y_test = tuple([Y_test, E_test])
    y_train_t = labtrans.fit_transform(Y_train, E_train)
    y_val_t = labtrans.transform(Y_val, E_val)

    val = (x_val, y_val_t)
    out_features = labtrans.out_features

    if len(x_train.shape) > 3:
        if ResNet:
            input_W = x_train.shape[-2]
            input_H = x_train.shape[-1]
            input_D = x_train.shape[2]
            model_depth = 50
            net = generate_model(model='resnet', model_depth=model_depth, input_W=input_W, input_H=input_H, input_D=input_D, \
                                        resnet_shortcut='B', pretrain_path=f'./models/resnet_{model_depth}.pth')
        else:
            net = models.Conv3D_torch(out_features, depth=depth)
    else:
        in_features = x_train.shape[1]
        if ResNet:
            net = models.ResNet1D_torch(num_classes=num_durations)
        else:
            net = tt.practical.MLPVanilla(in_features, dims, out_features, batch_norm,
                              dropout, output_bias=output_bias)
    model = MTLR(net, tt.optim.Adam, duration_index=labtrans.cuts)
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False
    log = model.fit(x_train, y_train_t, batch_size, epochs, callbacks, val_data=val)

    surv_test = model.predict_surv_df(x_test)
    surv_val = model.predict_surv_df(x_val)

    test_AUC, test_IBS, test_CI, test_surv_MSE = discrete_metrics(surv_test, y_train, y_test, plot=False, model_name="MTLR")
    val_AUC, val_IBS, val_CI, val_surv_MSE = discrete_metrics(surv_val,  y_train, y_val)

    return test_AUC, test_IBS, test_CI, test_surv_MSE, val_AUC, val_IBS, val_CI, val_surv_MSE, log
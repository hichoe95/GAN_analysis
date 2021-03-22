input_w = input_w.cuda(0).requires_grad_(True)
optim_vector = optimizer()
optim = torch.optim.Adam([input_w], lr = 0.005, betas=(0.9, 0.999), eps = 1e-8)
criterion = torch.nn.MSELoss(reduction = "mean")

vgg_16 = VGG16_perceptual().cuda(0)
upsample = nn.Upsample(scale_factor = 256/1024, mode = 'bilinear')

images = []

real_image_np = (real_image_np - real_image_np.min())/(real_image_np.max() - real_image_np.min())

################### Hook ##########################
feature_map = {}

def save_features(name):
    def fn(_, __, out):
#         print(out)
        if name not in feature_map:
            feature_map[name] = [out[0].detach().cpu().numpy()]
        else:
            feature_map[name].append(out[0].detach().cpu().numpy())
    return fn

optim_vector.module[0].layer5.register_forward_hook(save_features('layer5'))
optim_vector.module[0].layer7.register_forward_hook(save_features('layer7'))
optim_vector.module[0].layer11.register_forward_hook(save_features('layer11'))
optim_vector.module[0].layer15.register_forward_hook(save_features('layer15'))

######################################################

for i in range(1000):
    input_w18 = input_w.unsqueeze(1).expand(-1,18,512)
    output = optim_vector(input_w18)
    gen_image = output['image']
    
    gen_image = (gen_image - gen_image.min())/(gen_image.max() - gen_image.min())
    
    mse, per_loss = loss_fn(gen_image, torch.Tensor(real_image_np).cuda(0), criterion, upsample, vgg_16)
    
    loss = mse + per_loss
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    
    
    if i % 2 == 0:
#         plt.figure(figsize = (4,4))
        img = gen_image.detach().cpu().numpy().copy()
        img = adjust_dynamic_range(img, [img.min(), img.max()], [0,1])
#         print(type(img))
        images.append(img[0].transpose(1,2,0))
#         print(loss)
#         plt.imshow(img[0].transpose(1,2,0))
#         plt.axis('off')
#         plt.show()


input_w1 = input_w.unsqueeze(1).expand(-1, 18, 512)
# input_w1 = torch.tensor(input_w1, requires_grad=True)
input_w1 = input_w1.clone().detach().requires_grad_(True)
optim = torch.optim.Adam([input_w1], lr = 0.005, betas = (0.9, 0.999), eps = 1e-8)

for i in range(1000):
    output = optim_vector(input_w1)
    gen_image = output['image']
    
    gen_image = (gen_image - gen_image.min())/(gen_image.max() - gen_image.min())
    
    mse, per_loss = loss_fn(gen_image, torch.Tensor(real_image_np).cuda(0), criterion, upsample, vgg_16)
    
    loss = mse + per_loss
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i % 2 == 0:
#         plt.figure(figsize = (4,4))
        img = gen_image.detach().cpu().numpy().copy()
        img = adjust_dynamic_range(img, [img.min(), img.max()], [0,1])
#         print(type(img))
        images.append(img[0].transpose(1,2,0))
    
#     if i % 10 == 0:
#         plt.figure(figsize = (4,4))
#         img = gen_image.detach().cpu().numpy().copy()
#         img = adjust_dynamic_range(img, [img.min(), img.max()], [0,1])

#         print(loss)
#         plt.imshow(img[0].transpose(1,2,0))
#         plt.axis('off')
#         plt.show()
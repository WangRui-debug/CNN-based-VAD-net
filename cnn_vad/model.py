import torch
import torch.nn as nn


class VADnet(nn.Module):
    def __init__(self, in_ch=512, mid_ch=16, emb_ch=16, num_layers=4, negative_slope=0.1, dor=0.1):
        super(VADnet, self).__init__()

        assert num_layers > 2
        self.num_layers = num_layers
        self.eb = nn.Embedding(in_ch, emb_ch)
        self.f_idx = torch.arange(in_ch)

        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.lrelus = nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                ich, och = 1, mid_ch
            else:
                ich, och = mid_ch, mid_ch
            self.dropouts.append(nn.Dropout(p=dor))
            self.convs.append(nn.utils.weight_norm(nn.Conv1d(ich + emb_ch, och, 5, padding=(5 - 1) // 2)))
            self.lrelus.append(nn.LeakyReLU(negative_slope))
        self.end = nn.utils.weight_norm(nn.Conv1d(mid_ch, 1, 5, padding=(5 - 1) // 2))

    def forward(self, x):
        device = x.device
        n_batch, n_freq, n_frame = x
        out = x.reshape(n_batch * n_freq, 1, n_frame)
        c = self.f_idx.to(device, dtype=torch.int64).unsqueeze(0).repeat(n_batch, 1).flatten()
        c = self.eb(c).unsqueeze(2).repeat(1, 1, n_frame)  # (n_batch*n_freq, emb_ch, n_frame)
        for dropout, conv, lrelu in zip(self.dropouts, self.convs, self.lrelus):
            out = torch.cat((dropout(out), c), 1)
            out = lrelu(conv(out))
        out = self.end(out)  # (n_batch*n_freq, 1, n_frame)
        out = out.reshape(n_batch, n_freq, n_frame)  # Frequency-wise unnormalized log-probability sequence
        return out

# if __name__ == "__main__":
    #model = VADnet()
    #a = torch.randn(1, 1, 28, 28)
    #b = model(a)
    #print(b)
model = VADnet()
print(model)
#a = torch.randn(1, 28)
#b = model(a)
#print(b)
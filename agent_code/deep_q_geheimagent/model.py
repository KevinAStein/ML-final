
import torch
import torch.nn as nn 

# Dann definieren wir uns ein Geheimagenten Netzwerk.
class SecretNetwork(nn.Module):

    def __init__(self, input_shape):

        super(SecretNetwork, self).__init__() 

        # Hier passiert die Magie. Wir definieren uns die Layer.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 5, 2), # Das kann jetzt halt alles mögliche sein. Sind hier mal 2 Convolutional Layer.
                                   # Die Parameter sind: in_channels, out_channels, kernel_size, stride
                                   # Da dein Input das Spielfeld ist, ist in_channels=1, für rgb images zb in_channels=3.
                                   # Siehe https://pytorch.org/docs/stable/nn.html
            nn.ReLU(),
            nn.Conv2d(8, 32, 5, 2),
            nn.ReLU(),
        )

        n = self._get_conv_output(self.conv, input_shape)

        # Convolutional Layer verwendet man oft um Konzepte in Bildern zu verstehen. Um die Konzepte zu klassifizieren,
        # also Aktionen zu finden, kann man auch mal noch ein paar Linear Layer einbauen.
        self.linear = nn.Linear(n, 6) # Wieder ähnlich zu oben in_features, out_features. out_features=6, da es im Spiel 6 Aktionen gibt.


    # Einfach um die Anzahl der Elemente zu finden, die die Convolutional Layer ausgeben.
    def _get_conv_output(self, net, input_shape):

        input = torch.rand(input_shape).unsqueeze(0)
        input = torch.autograd.Variable(input)
        output = net(input)

        n = output.numel() # Zählt einfach die Anzahl der Elemente, die aus den Convolutional Layern kommt.

        return n

    # Hier wird beschrieben, wie wir durch das Netzwerk wandern.
    def forward(self, input):

        # Durch die Convolutional layer.
        output = self.conv(input)

        # Hier wird das Format von output in einen Vektor umgewandelt, damit wir es durch den Linear Layer füttern können.
        output = output.view(output.shape[0], -1)

        # Durch die Linear Layer.
        output = self.linear(output)

        return output
        


import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):

    def __init__(self, input_shape):
        super(ActorCritic, self).__init__()
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
        self.actor = nn.Linear(n, 6) # Wieder ähnlich zu oben in_features, out_features. out_features=6, da es im Spiel 6 Aktionen gibt.
        self.critic = nn.Linear(n, 1)
    # Einfach um die Anzahl der Elemente zu finden, die die Convolutional Layer ausgeben.
    def _get_conv_output(self, net, input_shape):

        input = torch.rand(input_shape).unsqueeze(0)
        input = torch.autograd.Variable(input)
        output = net(input)

        n = output.numel() # Zählt einfach die Anzahl der Elemente, die aus den Convolutional Layern kommt.
        return n
    
    # In a PyTorch model, you only have to define the forward pass. PyTorch computes the backwards pass for you!
    def forward(self, x):
        # Durch die Convolutional layer.
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return x
    
    # Only the Actor head
    def get_action_probs(self, x):
        x = self(x)
        x = self.actor(x)
        action_probs = F.softmax(x, dim=1)
        return action_probs
    
    # Only the Critic head
    def get_state_value(self, x):
        x = self(x)
        state_value = self.critic(x)
        return state_value
    
    # Both heads
    def evaluate_actions(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x), dim=1)
        state_values = self.critic(x)
        return action_probs, state_values
        

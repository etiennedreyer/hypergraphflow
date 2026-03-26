import torch
import torch.nn.functional as F
import numpy as np
from itertools import product, combinations


class SetGameData(torch.utils.data.IterableDataset):
    def __init__(self, 
                 set_size=3, 
                 num_feats=4, 
                 hand_size=12):

        self.num_feats = num_feats
        self.set_size = set_size
        self.hand_size = hand_size
        # Generate all cards
        self.deck = np.array(list(product(range(self.set_size), 
                                          repeat=self.num_feats)))

    @staticmethod
    def triplet_is_set(cards):
        return np.all(np.sum(cards, axis=0) % self.set_size == 0)

    def find_solutions(self, X):
        # Find all Sets
        solutions = [
            np.array(idxs) for idxs in combinations(range(len(X)), self.set_size)
            if self.triplet_is_set(X[list(idxs)])
        ]

        return np.array(solutions)

    def get_incidence_matrix(self, y):
        # Return an incidence matrix of shape (num_sets, hand_size)
        incidence_matrix = np.zeros((len(y), self.hand_size), dtype=bool)
        if len(y) == 0:
            return incidence_matrix
        edge_indices = np.arange(len(y)).repeat(self.set_size)
        node_indices = y.flatten()
        incidence_matrix[edge_indices, node_indices] = True
        return incidence_matrix

    def __iter__(self):
        while True:
            indices = np.random.choice(len(self.deck), self.hand_size, replace=False)
            X = self.deck[indices]
            y = self.find_solutions(X)
            I = self.get_incidence_matrix(y)

            X = torch.tensor(X, dtype=torch.float32)
            I = torch.tensor(I, dtype=torch.bool)

            yield X, I
 

### Number of nodes is fixed but number of edges can vary, 
### so we need to pad the incidence matrix to a fixed size
### Also, add binary indicator feature
def get_collate_fn(max_facets, one_hot_encoding=True, num_classes=3):
    def collate_fn(batch):
        points = []
        incidence = []
        for p, i in batch:
            if one_hot_encoding:
                ### E.g. p: (N, 4) with each feature in {0,1,2} -> (N, 12)
                p = F.one_hot(p.long(), num_classes=num_classes).float().reshape(p.size(0), -1)
            nf = i.size(0)
            if nf > max_facets:
                print(f"Warning: number of hyperedges {nf} exceeds maximum {max_facets}, truncating")
                i = i[:max_facets]
                nf = max_facets
            inc = torch.cat([i, torch.zeros(max_facets - nf, i.size(1))], dim=0)
            inc = torch.cat([inc, torch.zeros(max_facets, 1)], dim=1)
            inc[:nf, -1] = 1.
            incidence.append(inc)
            points.append(p)
        return torch.stack(points), torch.stack(incidence)
    return collate_fn


### Thanks Claude
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def VisualizeSetHandV2(X, I=None):
    """
    Visualize a SET hand of 12 cards.
    Each card has 4 features with 3 possible values each.
    
    Args:
        X: Array of shape (12, 4) with card features
        I: Optional incidence matrix of shape (num_sets, 12)
           Rows correspond to sets, columns to cards (0-11)
           True values indicate card membership in that set
    """
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('SET Hand - 12 Cards', fontsize=14, fontweight='bold')
    
    colors = ['red', 'green', 'blue']
    shape_funcs = {
        'o': lambda x, y, r: patches.Circle((x, y), r),
        '^': lambda x, y, r: patches.RegularPolygon((x, y), 3, radius=r, orientation=np.pi/2),
        's': lambda x, y, r: patches.Rectangle((x-r, y-r), 2*r, 2*r)
    }
    shapes = ['o', '^', 's']  # circle, triangle, square
    
    # Set colors for set membership boxes
    set_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, card in enumerate(X):
        ax = axes[i // 4, i % 4]
        
        count = int(card[0]) + 1  # 0->1, 1->2, 2->3
        color = colors[int(card[1])]
        shape_idx = int(card[2])
        shape = shapes[shape_idx]
        fill = int(card[3])  # 0=solid, 1=striped, 2=hollow
        
        # Position objects vertically in the card
        y_positions = np.linspace(0.25, 0.75, count)
        
        for y_pos in y_positions:
            patch = shape_funcs[shape](0.5, y_pos, 0.08)
            
            if fill == 0:  # Solid
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_linewidth(1.5)
            elif fill == 1:  # Striped
                patch.set_facecolor('white')
                patch.set_edgecolor(color)
                patch.set_linewidth(2)
                patch.set_hatch('///')
            else:  # Hollow (2)
                patch.set_facecolor('white')
                patch.set_edgecolor(color)
                patch.set_linewidth(2)
            
            ax.add_patch(patch)
        
        # Card styling
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#f0f0f0')
        
        # Add card border
        card_border = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(card_border)
    
    # Draw set membership boxes if incidence matrix provided
    if I is not None:
        # Convert to numpy if tensor
        if isinstance(I, torch.Tensor):
            I = I.numpy()
        
        num_sets = I.shape[0]
        
        # For each card, collect which sets it belongs to
        card_set_memberships = [[] for _ in range(12)]
        for set_idx in range(num_sets):
            card_indices = np.where(I[set_idx])[0]
            for card_idx in card_indices:
                card_set_memberships[card_idx].append(set_idx)
        
        # Draw boxes: largest first so smaller ones are visible on top
        for card_idx in range(12):
            if not card_set_memberships[card_idx]:
                continue
            
            # Sort by set index descending (largest/oldest sets first)
            sorted_sets = sorted(card_set_memberships[card_idx], reverse=True)
            
            ax_row = card_idx // 4
            ax_col = card_idx % 4
            ax = axes[ax_row, ax_col]
            
            for rank, set_idx in enumerate(sorted_sets):
                # Size increases with rank (larger boxes drawn first)
                box_size = 0.8 + rank * 0.03
                box_x = 0.5 - box_size / 2
                box_y = 0.5 - box_size / 2
                
                # Color cycles through set_colors
                color_idx = set_idx % len(set_colors)
                box_color = set_colors[color_idx]
                
                box = patches.Rectangle(
                    (box_x, box_y), box_size, box_size,
                    linewidth=2,
                    edgecolor=box_color,
                    facecolor='none',
                    alpha=0.8
                )
                ax.add_patch(box)
    
    plt.tight_layout()
    plt.show()
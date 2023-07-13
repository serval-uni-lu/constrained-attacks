# Design

Where do we inverse transform? 
Where do we fix feature types?

# How to update an attack to consider constraints

## Normalization

The model should always take as input, unscaled examples to simplify. 
If not, simply add a layer of normalization.

The attack always uses examples that are scaled (min/max + one hot encoded).

## Respecting the constraints

We describe from a high-level perspective how to adapt an attack to integrate 4 types of constraints:

- Relation constraints: 
	- integrate the minimization of the relation constraints in the objective (e.g. loss) of the attack.
	- Repair equality constraints
- Feature types:
    - One hot encoded: a forward and backward pass on the scaler is sufficient
    - Integer: Round values in the inverse direction of the perturbation to satisfy types and bounds constraints. (use constrained_attacks.utils.fix_types)
- Mutability constraints: 
    - Use a mask on the application of the perturbation such that only the mutable features are updated
    - Use a custom layer: this implies updating the EPS
- Bound constraints: clipping. Most attacks are already clipping the scaled value to [0, 1]

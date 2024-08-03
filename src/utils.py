import torch

def update_C(C: torch.Tensor, performance: torch.Tensor, barrier: float, measurement: str, style: str) -> torch.Tensor:
    """ 
    Updates C, an M x 1 tensor, based on the performance of each iteration's
    X-processes and the barrier value, as well as the measurement style
    """
    if measurement == 'WorstOf':
        P = performance.min(dim = 1, keepdim = True).values

    if measurement == 'BestOf':
        P = performance.max(dim = 1, keepdim = True).values

    if measurement == 'EqualWeighted':
        P = torch.sum(performance, dim = 1, keepdim = True)

    if measurement == 'Single':
        # NOTE: i'm assuming that for the style = 'Single' case, I can just go ahead and use 'performance' as if it were an integer
        # This may not work in the implementation, python may throw some errors - to check
        P = performance

    if style == 'up-and-out':
        breach = P >= barrier

    if style == 'up-and-in':
        breach = P <= barrier # NOTE: THIS SHOULD NOT BE USED - UP-AND-IN Options still confuse me

    # update C based on what has been breached
    updated_C = torch.logical_or(C, breach)

    return updated_C.int() # M x 1 tensor of 0s and 1s

def update_tFP(tFP: torch.Tensor, C: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """
    Based on this timestep's updated C, update tFP
    Recall that the logic for an element in tFP is :
        tFP(i) = t(i) * (1 - XTrig(i-1)) + tFP(i-1) * XTrig(i-1)
    
    where C takes place of XTrig

    meaning: tFP takes the value of the current timestep
             unless the corresponding element in C = 1,
             in which case it takes on the previous value of tFP.
             This will 'record' the timestep at which the breach was detected.

    Args:
        tFP: the current tFP tensor to update. Mx 1. This is tFPP(i-1)
        C : the updated C tensor for the current timestep. M x 1. Takse places of XTrig
        timestep: the current timestep's tensor.
                  derived from t[:, n+1, :], so an M x (N+1) x 1 => M x 1 tensor (value of current timestep is shared across all dimensions, the '1' element)    
    """
    # perform operations; all tensor of same dimension
    output = (timestep * (1.0 - C) # For elements where NO barrier breach, record current timestamp
               + tFP * C) # For elements WITH barrier breach, record same value as previous tFP (i.e. the timestep for whenever the barrier was breached)
    
    return output

def updated_XFP(XFP: torch.Tensor, C: torch.Tensor, X: torch.Tensor, D: int) -> torch.Tensor:
    """
    Based on this timestep's C, updated XFP
    Since XFP indicates the value of each X-process of each iteration at time t = tFP,
    XFP is an M x D tensor

    Args:
        XFP (M x D)
        C : the updated C tensor for the current timestep. M x 1. Indicates if barrier has been hit
            in this iteration/path
        X (M x D) : tensor showing the values of X-process for each iteration and each dimension 
                    at the current timestep
        D (int)   : indicates the number of dimensions being used. This is important when
                    enlarging C to allow for tensor algebra in the opeation with XFP and X
    """
    # Repeat C to be of the same dimensions as tensor X & XFP
    repeated_C = C.repeat(1, D)

    # Perform operation
    output = X * (1.0 - C) + XFP * C

    return output

def update_YFP(YFP: torch.Tensor, C: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Update to YFP, a tracker variable to see if this path's option has breached the barrier

    Args:
        YFP (M x 1 Tensor) : tracker variable for value of YFP at the barrier breach, if it has occurred
        C (M x 1 Tensor) : our indicator variable for if a barrier breach has occurred in this iteration
        Y (M x 1 Tensor) : the NN's prediction of the Y-value this timestep
    """
    # Perform operation
    output = Y * (1.0 - C) + YFP * C

    return output 


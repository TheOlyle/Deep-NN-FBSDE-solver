import torch

def check_breach(anchor: float, tA: torch.Tensor, tB: torch.Tensor) -> torch.Tensor:
    """ 
    given 2 tensors, checks element-wise to see if they breached a given 
    anchor
    e.g. tA has element 80, tB has corresponding element 120, and anchor = 120 => Breach!

    Used in FBSNN.loss_function to determine if there has been a breach
    tA is X0, tB is X1, and the anchor is FBSNN.domain_barrier
    """
    # Make sure the two twnsore are of equal shape
    assert tA.shape == tB.shape

    # Create a tensor of same dimensions, filled with the 'anchor' value
    # We'll use this to perform element-wise comparison
    tanchor = torch.full((tA.shape), anchor)

    # I want to check for EITHER type of barrier:
    # Barrier below X0 and breached from top->down by X1, or barrier above X0 and breached from bottom->up by X1
    C1 = ((tA <= tanchor) & (tanchor <= tB))
    C2 = ((tB <= tanchor) & (tanchor <= tA))

    # An element in either of the above can be true in order
    # for a 'Barrier Breach' to have occurred - domain breach
    C = torch.logical_or(C1, C2)

    return C

def update_XTrig(XTrig: torch.Tensor, breach_check: torch.Tensor, basket_measurement: str) -> torch.Tensor:
    """ 
    Based on this timestep's breach-check tensor (from check_breach)
    Updates XTrig accordingly
    Recall
        IF an element in XTrig is already 1, then it stays as 1
        IF an element in XTrig is 0, but a breach has occurred, update to 1
        IF an element in XTrig is 0, and a breach has not occurred, stay as 0
    
    NOTE: The logic of this NEEDS to change to account for different basket measurement styles.    
    """
    # For each element, check if either XTrig is already 1 OR if a breach has occurred
    updated_XTrig = torch.logical_or(XTrig, breach_check)

    # Convert to a 0 or 1 tensor
    updated_XTrig = updated_XTrig.int()

    return updated_XTrig

def update_tFP(tFP: torch.Tensor, XTrig: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """ 
    Based on this timestep's updated XTrig, update tFP
    Recall that the logic for an element in tFP is:
        tFP(i) = t(i) * (1-XTrig(i-1)) + tFP(i-1) * XTrig(i-1)
    
    Meaning: tFP takes the value of the current timestep
             unless the corresponding element in XTrig = 1, 
             in which case it takes on the previous value of tFP.
             This will 'record' the timestep at which the breach was detected.

    Args:
        tFP: the current tFP tensor to update. M x D. This is tFP(i-1)
        XTrig: the updated XTrig tensor for the current timestep. M x D
        current_t: the current timestep's tensor.
                   derived from t[:, n+1, :], so an M x (N+1) x 1 => M x 1 tensor (value of current timestep is shared across all dimensions, the '1' element)
    """
    # timestep is based on t0, which is of shape M x 1
    # to perform element-wise operations, I need to convert it to same shape as
    # XTrig, which is M x D
    timestep = timestep.expand((XTrig.shape))

    # Perform operations
    output = timestep * (1.0 - XTrig) + tFP * XTrig

    return output

def update_XFP(XFP: torch.Tensor, XTrig: torch.tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Updated XFP - need to write more expanis docstring
    """
    # Perform operation
    output = X * (1.0 - XTrig) + XFP * XTrig

    return output

def update_YFP(YFP: torch.Tensor, XTrig: torch.Tensor, Y: torch.Tensor, basket_measurement: str) -> torch.Tensor:
    """
    Update to YFP, a tracker variable to see if this path's option has breached the barrier
    Recall for a single-dimension case:
        YFP = Y * (1 - XTrig) + YFP * XTrig
    Meaning: if this option's single X-dimension has breached the barrier, YFP is updated because XTrig is updated.
    For basket-options with barriers, it's more complex:
        If Worst-Of:
            The X-dimension with the worst relative performance to its initial value (X at t = 0) is relevant 
            to determine if there has been a barrier breach.
        If Best-Of:
            The X-dimension with the best relative performance to its initial value (X at t = 0) is relevant
            to determine if there has been a barrier breach.
        If Equal-Weighted:
            The performance of all of the X-dimensions relative to initial value (X at t = 0) is measured and 
            averaged. This will determine if there has been a barrier breach.

    NOTE: I NEED TO RETHINK THIS LOGIC to incorporate FBSNN.basket_measurement
    
    Args:
        YFP (torch.Tensor) : YFP tensor, which is M x 1
    """
    # Y1 from self.net_u() is an M x 1 tensor
    # to perform element-wise operations, I need to convert it to the same shape as
    # XTrig, which is M X D
    Y = Y.expand((XTrig.shape))

    # Perform operation
    output = Y * (1.0 - XTrig) + YFP * XTrig

    return output

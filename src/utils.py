import torch

def check_breach(anchor: float, tA: torch.Tensor, tB: torch.Tensor) -> bool:
    """ 
    given 2 tensors, checks element-wise to see if they breached a given 
    anchor
    e.g. tA has element 80, tB has corresponding element 120, and anchor = 120 => Breach!

    Used in FBSNN.loss_function to determine if there has been a breach
    """
    # Make sure the two twnsore are of equal shape
    assert tA.Shape == tB.Shape

    # Create a tensor of same dimensions, filled with the 'anchor' value
    # We'll use this to perform element-wise comparison
    tanchor = torch.full((tA.Shape), anchor)

    # I want to check for EITHER type of barrier:
    # Barrier below X0 and breached from top->down by X1, or barrier above X0 and breached from bottom->up by X1
    C1 = ((tA <= tanchor) & (tanchor <= tB))
    C2 = ((tB <= tanchor) & (tanchor <= tA))

    # An element in either of the above can be true in order
    # for a 'Barrier Breach' to have occurred - domain breach
    C = torch.logical_or(C1, C2)

    return C

def update_XTrig(XTrig: torch.Tensor, breach_check: torch.Tensor) -> torch.Tensor:
    """ 
    Based on this timestep's breach-check tensor (from check_breach)
    Updates XTrig accordingly
    Recall
        IF an element in XTrig is already 1, then it stays as 1
        IF an element in XTrig is 0, but a breach has occurred, update to 1
        IF an element in XTrig is 0, and a breach has not occurred, stay as 0
    """
    # For each element, check if either XTrig is already 1 OR if a breach has occurred
    updated_XTrig = torch.logical_or(XTrig, breach_check)

    # Convert to a 0 or 1 tensor
    updated_XTrig = updated_XTrig.int()

    return updated_XTrig

from scenarios.overcooked.common import MacroCommand, Position, PotState


def test_macro_command_parse():
    cmd = MacroCommand.parse("pickup ingredient_0")
    assert cmd.verb == "pickup"
    assert cmd.target == "ingredient_0"


def test_macro_command_parse_no_target():
    cmd = MacroCommand.parse("deliver")
    assert cmd.verb == "deliver"
    assert cmd.target == ""


def test_pot_state_lifecycle():
    p = PotState(pos=Position(0, 0))
    assert p.is_idle
    p.ingredients = 3
    p.cooking_timer = 5
    p.cooking = True
    assert p.is_cooking
    p.cooking_timer = 0
    p.cooking = False
    p.cooked = True
    assert p.is_ready

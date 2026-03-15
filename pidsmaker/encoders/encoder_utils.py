def get_encoder_methods(used_methods):
    return [method.strip() for method in used_methods.replace("-", ",").split(",") if method.strip()]


def encoder_has_method(used_methods, method_name):
    return method_name in get_encoder_methods(used_methods)


def encoder_uses_tgn(used_methods):
    methods = get_encoder_methods(used_methods)
    return "tgn" in methods or "early_fusion" in methods

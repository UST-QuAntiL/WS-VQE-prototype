import json
import pickle

def store_results(fname, problem_instance, ref_eigenvalue, ref_eigenstate, approx_eigenstate, approx_ratio, _variants):
    # adapt some values so they can be stored as json
    # remove other values, that are not needed
    for variant in _variants:
        variant["kwargs"]["optimizer"] = {"name": type(variant["kwargs"]["optimizer"]).__name__, "options": variant["kwargs"]["optimizer"]._options}
        variant["result"].optimal_circuit = None
        variant["result"].optimal_parameters = None
        variant["result"].optimal_point = list(variant["result"].optimal_point)
        variant["result"].optimizer_result = None
        variant["result"] = eval(str(variant["result"]))
        if "initial_point" in variant["kwargs"].keys():
            variant["kwargs"]["initial_point"] = str(variant["kwargs"]["initial_point"])
    results_json = {
                        "problem instance": problem_instance.dumps().decode("ISO-8859-1"),
                        "ref eigenvalue": ref_eigenvalue,
                        "ref eigenstate": str(list(ref_eigenstate)),
                        "approx. eigenstate": str(list(approx_eigenstate)),
                        "approx. ratio": approx_ratio,
                        "results": _variants
                   }
    with open(fname, "a") as outfile:
        json.dump(results_json, outfile)
        outfile.write("\n")
    return problem_instance, ref_eigenvalue, ref_eigenstate, approx_eigenstate, approx_ratio, _variants

def load_results(fname):
    # nothing is secure here, handle with care
    # restores the values from file, reverting some of the actions above
    with open(fname, "r") as infile:
        lines = infile.readlines()

    results = []

    for i, line in enumerate(lines):
        #try:
        result_json = json.loads(line)
        #except:
            #print("line", i, line[:100])
        result_json["problem instance"] = pickle.loads(result_json["problem instance"].encode("ISO-8859-1"))
        result_json["ref eigenstate"] = eval(result_json["ref eigenstate"])
        result_json["approx. eigenstate"] = eval(result_json["approx. eigenstate"])
        for variant in result_json["results"]:
            if "initial_point" in variant["kwargs"]:
                variant["kwargs"]["initial_point"] = eval(variant["kwargs"]["initial_point"])
            variant["values"] = {int(float(c)): v for c, v in variant["values"].items()}

        results.append(result_json)

    return results

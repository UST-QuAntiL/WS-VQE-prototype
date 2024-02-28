# WS-VQE-prototype

This repository contains the prototypical implementation used to generate results presented in our publication **"[Warm-Starting the VQE with Approximate Complex Amplitude Encoding](https://arxiv.org/abs/2402.17378)"** [1]

The main contents of this repository are as follows.

[fidelity_estimation.py](fidelity_estimation.py) contains the functionality to prepare measurement circuits for the fidelity estimation based on classical shadows with Clifford measurements, see also [2,3].

[approximate_encoding.py](approximate_encoding.py) implements Approximate Complex Amplitude Encoding (ACAE) based on the estimated fidelities, see [3].

[compare_vqe_variants.py](compare_vqe_variants.py) compares different variants of the VQE with and without warm-starts based on ACAE with different optimizer settings as further explained in our paper [1]. It contains functions to run these variants for multiple problem instances, store information on the progress of each variant, and evaluate these results.

[utils](utils) provides functionalities to generate the problem instances, obtain approximate solutions by means of the inverse power method, handle the storage of results, and a wrapper for the optimizer.

[results](results) contains the results generated with the implementation and presented in the publication. The evaluation is done with the appropriate function provided in [compare_vqe_variants.py](compare_vqe_variants.py).

### References
[1] Truger, Felix, et al. "Warm-Starting the VQE with Approximate Complex Amplitude Encoding." arXiv:2402.17378 (2024)

[2] Huang, Hsin-Yuan, Richard Kueng, and John Preskill. "Predicting many properties of a quantum system from very few measurements." Nature Physics 16.10 (2020): 1050-1057.

[3] Mitsuda, Naoki, et al. "Approximate complex amplitude encoding algorithm and its application to classification problem in financial operations." arXiv:2211.13039 (2022).

### Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

### Haftungsausschluss

Dies ist ein Forschungsprototyp. Die Haftung für entgangenen Gewinn, Produktionsausfall, Betriebsunterbrechung, entgangene Nutzungen, Verlust von Daten und Informationen, Finanzierungsaufwendungen sowie sonstige Vermögens- und Folgeschäden ist, außer in Fällen von grober Fahrlässigkeit, Vorsatz und Personenschäden, ausgeschlossen.

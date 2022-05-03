<div id="top"></div>

# ABIDES: Agent-Based Interactive Discrete Event Simulation environment

<!-- TABLE OF CONTENTS -->
<ol>
  <li>
    <a href="#about-the-project">About The Project</a>
  </li>
  <li><a href="#citing-abides">Citing ABIDES</a></li>
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#installation">Installation</a></li>
    </ul>
  </li>
  <li>
    <a href="#usage-regular">Usage (regular)</a>
    <ul>
      <li><a href="#using-a-python-script">Using a Python Script</a></li>
      <li><a href="#using-the-abides-command">Using the `abides` Command</a></li>
    </ul>
  </li>
  <li><a href="#usage-gym">Usage (Gym)</a></li>
  <li><a href="#default-available-markets-configurations">Default Available Markets Configurations</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>

<!-- ABOUT THE PROJECT -->
## About The Project

ABIDES (Agent Based Interactive Discrete Event Simulator) is a general purpose multi-agent discrete event simulator. Agents exclusively communicate through an advanced messaging system that supports latency models.

The project is currently broken down into 3 parts: ABIDES-Core, ABIDES-Markets and ABIDES-Gym.

* ABIDES-Core: Core general purpose simulator that be used as a base to build simulations of various systems.
* ABIDES-Markets: Extension of ABIDES-Core to financial markets. Contains implementation of an exchange mimicking NASDAQ, stylised trading agents and configurations.
* ABIDES-Gym: Extra layer to wrap the simulator into an OpenAI Gym environment for reinforcement learning use. 2 ready to use trading environments available. Possibility to build other financial markets environments easily.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CITING -->
## Citing ABIDES

[ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation and Application to Financial Markets](https://arxiv.org/pdf/2110.14771.pdf) or use
the following BibTeX:

```
@misc{amrouni2021abidesgym,
      title={ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation and Application to Financial Markets}, 
      author={Selim Amrouni and Aymeric Moulin and Jared Vann and Svitlana Vyetrenko and Tucker Balch and Manuela Veloso},
      year={2021},
      eprint={2110.14771},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```

[ABIDES: Towards High-Fidelity Market Simulation for AI Research](https://arxiv.org/abs/1904.12066)
or by using the following BibTeX:

```
@misc{byrd2019abides,
      title={ABIDES: Towards High-Fidelity Market Simulation for AI Research}, 
      author={David Byrd and Maria Hybinette and Tucker Hybinette Balch},
      year={2019},
      eprint={1904.12066},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
### Installation

1. Download the ABIDES source code, either directly from GitHub or with git:

    ```bash
    git clone https://github.com/jpmorganchase/abides-jpmc-public
    ```

    **Note:** The latest stable version is contained within the `main` branch.

2. Run the install script to install the ABIDES packages and their dependencies:

    ```
    sh install.sh
    ```


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage (regular)
Regular ABIDES simulations can be run either directly in python or through the command line

_For more examples, please refer to the [Documentation](https://example.com)_

### Using a Python Script:

```python
from abides_markets.configs import rmsc04
from abides_core import abides

config_state = rmsc04.build_config(seed = 0, end_time = '10:00:00')
end_state = abides.run(config_state)
```
<p align="right">(<a href="#top">back to top</a>)</p>

### Using the abides Command:

The config can be loaded and the simulation run using the `abides`
command in the terminal (from directory root):

```
$ abides abides-markets/abides_markets/configs/rmsc04.py --end_time "10:00:00"
```

The first argument is a path to a valid ABIDES configuration file.

Any further arguments are optional and can be used to overwrite any parameters
in the config file.

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage (Gym)
ABIDES can also be run through a Gym interface using ABIDES-Gym environments.

```python
import gym
import abides_gym

env = gym.make(
    "markets-daily_investor-v0",
    background_config="rmsc04",
)

env.seed(0)
initial_state = env.reset()
for i in range(5):
    state, reward, done, info = env.step(0)
```

## Default Available Markets Configurations

ABIDES currently has the following available background Market Simulation Configuration:

* RMSC03: 1 Exchange Agent, 1 POV Market Maker Agent, 100 Value Agents, 25 Momentum Agents, 5000 Noise Agents
 
* RMSC04: 1 Exchange Agent, 2 Market Maker Agents, 102 Value Agents, 12 Momentum Agents, 1000  Noise Agents

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

TODO: add information about JPMC contribution agreement

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License
Distributed under the BSD 3-Clause "New" or "Revised" License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
ABIDES was originally developed by David Byrd and Tucker Balch: https://github.com/abides-sim/abides
ABIDES is currently developed and maintained by [Jared Vann](https://github.com/jaredvann) (aka @jaredvann), [Selim Amrouni](https://github.com/selimamrouni) (aka @selimamrouni), and [Aymeric Moulin](https://github.com/AymericCAMoulin) (@AymericCAMoulin).
**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.

<p align="right">(<a href="#top">back to top</a>)</p>

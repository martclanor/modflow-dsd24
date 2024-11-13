# Setting up Extended MODFLOW and class materials

Instructions are provided here for installing the extended version of MODFLOW (with parallel and netcdf support), and for getting access to class materials and preparing the Python environment so you can follow along with the demos and exercises. We have three different options to prepare your machine, listed here in order of maturity:

- If you have WSL successfully installed, you can continue and execute steps 1,2,3,4 below. 

- Alternatively, in case you don't have WSL up and running or prefer to work in a Windows CMD shell instead, simply replace step 1 with 1A. 

- Alternatively, for native Linux or MacOS machines, replace step 1 with 1B.


## 1. Installing WSL
_Note that you need elevated privileges and sometimes your IT administrator to complete this step_

On a Windows machine it is relatively easy to get Extended MODFLOW compiled and running in a WSL Ubuntu virtual machine.

Install a latest version of Ubuntu.
```
  wsl --install -d Ubuntu-22.04
```

You will be asked to provide a username and password. You'll need to remember this information for some future `sudo` operations. 

Alternatively, you can install Ubuntu-22.04 from the Microsoft Store. In that case, you (or your administrator) might need to activate WSL explicitly:

1.	Open Windows 10 Settings app.
2.	Select Apps.
3.	Click Programs and Features under the Related settings section on the right.
4.	Under the Programs and Features page, click Turn Windows features on or off on the left panel.
5.	Scroll down and enable Windows Subsystem for Linux.

With Ubuntu/Linux available in your WSL, you can go to **step 2**.

## 1A. Windows native installation (Alternative)
Recently a nightly-build has become available for extended MODFLOW on Windows. This workflow installs the right version of MODFLOW automatically as opposed to building your own from sources. The details are part of the installation script below. Just follow the instructions starting from **step 2**.

## 1B. Linux or MacOS (Alternative)
There usually is no problem running the instructions directly on Linux or MacOS. You can proceed from **step 2**.

## 2. Clone the class repo

When typing "Ubuntu" in your Windows Search box, the installed version should become visible in the Apps section. Click to start a terminal window. Alternatively, start a CMD shell or a terminal in your favorite MacOS or Linux environment.

In the terminal, clone the class repo using the following command:

```
git clone https://github.com/jdhughes-usgs/modflow-dsd24.git
```

## 3. Setting up extended MODFLOW 6 and pixi environment

In order to set up extended MODFLOW 6, install pixi:

```
curl -fsSL https://pixi.sh/install.sh | bash
```

Additional information on installing pixi is available at [https://pixi.sh/dev/](https://pixi.sh/dev/). Next execute:

```
cd modflow-dsd24
pixi run install
```

This will take a bit of time, somewhere in between 15 minutes and an hour. A successful installation sequence concludes with the message:

```
Successful testing of pixi environment and MODFLOW 6
```

## 4. Start Jupyter Lab

Jupyter Lab can be started by executing:

```
./pixi run jupyter
```
and (when using WSL) clicking or copying the link at the end of the message:

```
 To access the server, open this file in a browser:
      file:///home/user/.local/share/jupyter/...
 Or copy and paste one of these URLs:
      http://localhost:8888/lab?token=...
      http://127.0.0.1:8888/lab?token=...
```

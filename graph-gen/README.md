# Graph generator

## Overview
This software repository contains an experimental software implementation of algorithms for solving a set of pattern-detection problems in temporal graphs. The software is written in C programming language.

This version of the source code is realeased for SDM - 2020 submission titled "Pattern detection in large temporal graphs using algebraic fingerprints".

## License
The source code is subject to MIT license.

## Compilation
The source code is configured for a gcc build. Other builds are possible but it might require manual configuration of the 'Makefile'.

Use GNU make to build the software. Check 'Makefile' for more details.

`make clean all`

## Using the software
Usage:

`./graph-gen <type> <arguments>`

Arguments:

	regular         <n> <d> <k> <t> <c> <nt> <seed>           (with 1 <= k <= n and n*d even)
	powlaw          <n> <d> <al> <w> <k> <t> <c> <nt> <seed>  (with al < 0.0, 2 <= w <= n, 1 <= k <= n)
	regular-rainbow <n> <d> <k> <t> <c> <nt> <seed>           (with 1 <= k <= n and n*d even)
	powlaw-rainbow  <n> <d> <al> <w> <k> <t> <c> <nt> <seed>  (with al < 0.0, 2 <= w <= n, 1 <= k <= n)

		n    - number of nodes
		d    - degree
		k    - multi-set size
		t    - maximum timestamp
		c    - number of colors
		nt   - number of targets
		seed - seed for random graph instance
		al   - alpha (power-law graphs)
		w    - weight (power-law graphs)

## Example

`$./graph-gen regular 1000 20 5 100 5 10 12345`
`$./graph-gen powlaw-rainbow 1000 20 -0.5 100 5 100 5 10 12345`

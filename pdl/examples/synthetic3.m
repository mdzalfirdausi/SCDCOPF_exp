function mpc = synthetic3
% Synthetic smoke-test fixture. This is NOT a PGLib or paper benchmark case.
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
10 3 80 0 0 0 1 1 0 1 1 1.1 .9;
20 1 70 0 0 0 1 1 0 1 1 1.1 .9;
30 1 50 0 0 0 1 1 0 1 1 1.1 .9;
];
mpc.gen = [
10 0 0 0 0 1 100 1 200 0;
20 0 0 0 0 1 100 1 200 0;
30 0 0 0 0 1 100 1 100 0;
];
mpc.gencost = [
2 0 0 2 10 0;
2 0 0 2 20 0;
2 0 0 2 30 0;
];
mpc.branch = [
10 20 0 .2 0 40 0 0 0 0 1 -360 360;
20 30 0 .3 0 40 0 0 0 0 1 -360 360;
10 30 0 .4 0 40 0 0 0 0 1 -360 360;
];

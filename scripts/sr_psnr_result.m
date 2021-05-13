wrong = dlmread('../result/wrong/psnr.txt');
base = dlmread('../result/base/psnr.txt');
finetune = dlmread('../result/finetune/psnr.txt');


ab_R = dlmread('../data/ab_R.txt');
ab_G = dlmread('../data/ab_G.txt');
ab_B = dlmread('../data/ab_B.txt');

figure;
hold on;
plot(ab_R(1,1:2:end),'r-*');
plot(ab_G(1,1:2:end),'g-*');
plot(ab_B(1,1:2:end),'b-*');

plot(ab_R(1,2:2:end)+0.002,'r-^');
plot(ab_G(1,2:2:end)+0.004,'g-^');
plot(ab_B(1,2:2:end),'b-^');

legend('\mu_R','\mu_G','\mu_B','\lambda_R','\lambda_G','\lambda_B');
xlabel('stage');
grid on;

nBin = 10;
figure;
hold on;
h = histogram(wrong,nBin);
wrongp = histcounts(wrong,nBin,'Normalization','pdf');
wrongCenters = h.BinEdges + (h.BinWidth/2);

h = histogram(base,nBin);
basep = histcounts(base,nBin,'Normalization','pdf');
baseCenters = h.BinEdges + (h.BinWidth/2);

h = histogram(finetune,nBin);
finetunep = histcounts(finetune,nBin,'Normalization','pdf');
finetuneCenters = h.BinEdges + (h.BinWidth/2);

figure;
hold on;
plot(wrongCenters(1:end-1), wrongp, 'r-')
plot(baseCenters(1:end-1), basep, 'g-')
plot(finetuneCenters(1:end-1), finetunep, 'b-')
grid on;
xlabel('PSNR');
ylabel('Density');
legend('Gong Jins','base','finetune');
% histogram(wrong);
% histogram(base);
% histogram(finetune);




# Lecture9Lab6
Lab 6: Mini Project – Image Pipeline for a Mountain Landscape
Problem definition

Image: mountain.jpg, a grayscale landscape photo of a mountain range, 852×1280 pixels.

Problem: Landscape photos often have low contrast (haze, flat lighting) and sensor noise, which make the terrain hard to see. The goal is to clean and enhance the image and extract the main structure of the scene: the ridgelines, the skyline and the terrain contours. A pipeline like this could be the first step in skyline detection, terrain mapping or image matching.

Pipeline
Stage	Operation	MATLAB	DSP concept	Lab
1. Load	RGB → grayscale, convert to double	rgb2gray, im2double	Image as a 2D signal, quantization	1
2. Denoise	3×3 median filter	medfilt2	Non-linear filtering, removing impulse noise	4
3. Enhance	Linear contrast stretch	imadjust(…,[0.2 0.8],[0 1])	Amplitude remapping, histogram stretching	1
4. Features	Canny edge detection	edge(…,'Canny')	Gradient = high-pass filter; edges = high frequencies	2, 5
5. Frequency	Ideal circular low-pass, D0 = 60	fft2, fftshift, mask, ifft2	Convolution theorem, frequency-domain filtering	3
6. Visualize	Side-by-side montage	montage	Comparing the stages	—
How each stage relates to DSP operations
Load and convert: the image becomes a 2D discrete signal I[m,n], sampled on an 852×1280 grid. im2double maps 8-bit levels (0–255) to [0, 1], so later arithmetic does not overflow or lose precision.
Median filter: a non-linear filter that replaces each sample with the median of its 3×3 neighbourhood. It removes impulse noise (isolated bright or dark pixels) and keeps edges sharp, unlike a mean filter (Lab 4). It goes before enhancement, because contrast stretching would otherwise amplify the noise too.
Contrast stretching: a linear point operation that maps input values 0.2–0.8 onto the full range 0–1. The histogram spreads out, so ridges, snow and shadows separate more clearly. Values outside the range are clipped, so the darkest shadows and brightest sky lose some detail.
Canny edges: Gaussian smoothing (low-pass), then gradient computation (high-pass), non-maximum suppression and hysteresis thresholding with low = 0.1 and high = 0.25. The output is a binary map of thin, connected contours: the high-frequency structure of the scene, such as ridgelines and rock boundaries.
Frequency-domain low-pass: the FFT turns the image into its spectrum. Multiplying by a circular mask H (radius 60) keeps only low frequencies, which by the convolution theorem is the same as convolving with a 2D sinc kernel. The result keeps the large shapes (mountains, sky, valleys) and removes fine texture. The mask has a hard cutoff, so faint ringing (Gibbs phenomenon) can appear next to strong edges such as the skyline.
Visualization: the montage shows the progression from raw image to cleaned, enhanced, structural (edges) and smoothed (low-pass) versions.
Results
Denoised: looks almost the same as the original at this scale. The median filter removes only isolated noise pixels and keeps the terrain detail.
Enhanced: clearly more contrast. The mountain silhouette and the light/dark structure of the slopes stand out, and dark areas get darker because of clipping.
Edges: in the montage, only scattered dots are visible, because each panel is shrunk to about 1/10 of its width and thin 1-pixel lines are lost. At full size, the ridgelines and terrain contours show up as connected lines. The thresholds [0.1 0.25] are also fairly strict, so weak edges are dropped.
LP result: a smooth, blurred version that keeps the overall shape of the mountains and loses fine texture, with possible faint ringing along the skyline.
Limitations and improvements
Limitation	Improvement
Ideal LP mask (hard cutoff) causes ringing	Use a Gaussian mask: H = exp(−D²/2σ²) (Lab 3)
Fixed stretch [0.2 0.8] clips shadows and highlights; one global setting for the whole scene	stretchlim for automatic limits, or CLAHE (adapthisteq), which adapts to local regions such as sky vs foreground
Canny thresholds chosen by hand	Lower or automatic thresholds (edge(I,'Canny') with no thresholds), or tune σ
Edges lost in the montage display	Show the edge map at full size, or thicken it with imdilate for display only
meshgrid(-N/2:N/2-1, …) only works for even image sizes	Use -floor(N/2):ceil(N/2)-1 (as in Lab 3) so any size works
No numbers to measure improvement	Report contrast (std of intensity), edge-pixel count, and before/after histograms
The median filter does little if the photo is already clean	Could be replaced by an edge-preserving smoother (imbilatfilt) to suppress texture before edge detection
Improved script (optional; addresses the limitations above)
matlab
%% Lab 6: Mini Project – Mountain Landscape Pipeline (improved)
close all; clear; clc;
I = im2double(rgb2gray(imread('mountain.jpg')));
[M,N] = size(I);

% 2) Denoise
I_filt = medfilt2(I,[3 3]);

% 3) Enhance: fixed stretch vs adaptive CLAHE
I_enh   = imadjust(I_filt,[0.2 0.8],[0 1]);
I_clahe = adapthisteq(I_filt,'ClipLimit',0.01);

% 4) Edges (slightly lower thresholds)
edges = edge(I_clahe,'Canny',[0.05 0.15]);

% 5) Frequency domain: ideal vs Gaussian low-pass
F = fftshift(fft2(I_enh));
[u,v] = meshgrid(-floor(N/2):ceil(N/2)-1, -floor(M/2):ceil(M/2)-1);
D = sqrt(u.^2 + v.^2);
H_ideal = double(D < 60);
H_gauss = exp(-(D.^2)/(2*40^2));
I_lp_ideal = real(ifft2(ifftshift(F.*H_ideal)));
I_lp_gauss = real(ifft2(ifftshift(F.*H_gauss)));

% 6) Visualization
figure; montage({I, I_filt, I_enh, I_clahe},'Size',[2 2]);
title('Original | Denoised | Stretched | CLAHE');

figure; imshow(imdilate(edges, strel('disk',1)));   % thicken for display only
title(sprintf('Canny edges (full size) – %d edge pixels', nnz(edges)));

figure; montage({I_lp_ideal, I_lp_gauss},'Size',[1 2]);
title('Ideal LP (ringing) | Gaussian LP (smooth)');

figure;
subplot(1,3,1); imhist(I);       title('Histogram: original');
subplot(1,3,2); imhist(I_enh);   title('Histogram: stretched');
subplot(1,3,3); imhist(I_clahe); title('Histogram: CLAHE');

% Simple numbers for the report
fprintf('Contrast (std): original=%.3f | stretched=%.3f | CLAHE=%.3f\n', ...
        std(I(:)), std(I_enh(:)), std(I_clahe(:)));

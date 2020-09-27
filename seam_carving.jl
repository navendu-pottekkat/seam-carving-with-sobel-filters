### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 4736c4a6-007f-11eb-1587-f773de9a54d1
begin
	using Pkg
	Pkg.activate(tempname())
end

# ╔═╡ 643ec760-007f-11eb-02b6-879b064093a1
begin
	using Images
	using ImageMagick
	using Statistics
	using LinearAlgebra
	using ImageFiltering
	using PlutoUI
end

# ╔═╡ 2eff7c8e-007f-11eb-337c-71e24bbe5b1d
md"""
# Seam Carving using Sobel filters
"""

# ╔═╡ 7bda3106-0080-11eb-3880-f911043ee036
md"""
Slide to reduce the size of the image using seam carving.
"""

# ╔═╡ 60a22fea-007f-11eb-0b85-61310e25b777
Pkg.add(["Images",
		 "ImageMagick",
		 "PlutoUI",
		 "Hyperscript",
		 "ImageFiltering"])

# ╔═╡ 9739002c-007f-11eb-34c6-cddee3fba7e0
	function show_colored_array(array)
		pos_color = RGB(0.36, 0.82, 0.8)
		neg_color = RGB(0.99, 0.18, 0.13)
		to_rgb(x) = max(x, 0) * pos_color + max(-x, 0) * neg_color
		to_rgb.(array) / maximum(abs.(array))
	end

# ╔═╡ 3badb08c-0080-11eb-094b-83d4ee955eb7
brightness(c::AbstractRGB) = 0.3 * c.r + 0.59 * c.g + 0.11 * c.b

# ╔═╡ 9eb27932-007f-11eb-1eb8-a95c26681b65
function convolve(M, kernel)
    height, width = size(kernel)
    
    half_height = height ÷ 2
    half_width = width ÷ 2
    
    new_image = similar(M)
	
	m, n = size(M)
    @inbounds for i in 1:m
        for j in 1:n
			accumulator = 0 * M[1, 1]
			for k in -half_height:-half_height + height - 1
				for l in -half_width:-half_width + width - 1
					Mi = i - k
					Mj = j - l
					if Mi < 1
						Mi = 1
					elseif Mi > m
						Mi = m
					end
					if Mj < 1
						Mj = 1
					elseif Mj > n
						Mj = n
					end
					
					accumulator += kernel[k, l] * M[Mi, Mj]
				end
			end
			new_image[i, j] = accumulator
        end
    end
    
    return new_image
end

# ╔═╡ a6ea06ba-007f-11eb-3b29-895a03842e8d
function hbox(x, y, gap=16; sy=size(y), sx=size(x))
	w,h = (max(sx[1], sy[1]),
		   gap + sx[2] + sy[2])
	
	slate = fill(RGB(1,1,1), w,h)
	slate[1:size(x,1), 1:size(x,2)] .= RGB.(x)
	slate[1:size(y,1), size(x,2) + gap .+ (1:size(y,2))] .= RGB.(y)
	slate
end

# ╔═╡ b0f4f2aa-007f-11eb-2732-9330dd587d4b
function edgeness(img)
	Sy, Sx = Kernel.sobel()
	b = brightness.(img)

	∇y = convolve(b, Sy)
	∇x = convolve(b, Sx)

	sqrt.(∇x.^2 + ∇y.^2)
end

# ╔═╡ b57ba6e0-007f-11eb-117f-cbf65a98ca47
function least_edgy(E)
	least_E = zeros(size(E))
	dirs = zeros(Int, size(E))
	least_E[end, :] .= E[end, :] 
	m, n = size(E)
	for i in m-1:-1:1
		for j in 1:n
			j1, j2 = max(1, j-1), min(j+1, n)
			e, dir = findmin(least_E[i+1, j1:j2])
			least_E[i,j] += e
			least_E[i,j] += E[i,j]
			dirs[i, j] = (-1,0,1)[dir + (j==1)]
		end
	end
	least_E, dirs
end

# ╔═╡ bdd0c206-007f-11eb-2030-cbb33969b289
function get_seam_at(dirs, j)
	m = size(dirs, 1)
	js = fill(0, m)
	js[1] = j
	for i=2:m
		js[i] = js[i-1] + dirs[i-1, js[i-1]]
	end
	tuple.(1:m, js)
end

# ╔═╡ c667e778-007f-11eb-25ca-c55d58c6bab9
function mark_path(img, path)
	img′ = copy(img)
	m = size(img, 2)
	for (i, j) in path

		for j′ in j-1:j+1
			img′[i, clamp(j′, 1, m)] = RGB(1,0,1)
		end
	end
	img′
end

# ╔═╡ c7396bc2-007f-11eb-13d6-61659dbccbdc
function rm_path(img, path)
	img′ = img[:, 1:end-1] 
	for (i, j) in path
		img′[i, 1:j-1] .= img[i, 1:j-1]
		img′[i, j:end] .= img[i, j+1:end]
	end
	img′
end

# ╔═╡ cb0c5b4c-007f-11eb-2974-23387d6d1038
function shrink_n(img, n)
	imgs = []
	marked_imgs = []

	e = edgeness(img)
	for i=1:n
		least_E, dirs = least_edgy(e)
		_, min_j = findmin(@view least_E[1, :])
		seam = get_seam_at(dirs, min_j)
		img = rm_path(img, seam)

		e = rm_path(e, seam)

 		push!(imgs, img)
 		push!(marked_imgs, mark_path(img, seam))
	end
	imgs, marked_imgs
end

# ╔═╡ 0a370f74-0080-11eb-3fce-2f855eeb0058
img = load(download("https://cdn.shortpixel.ai/spai/w_1086+q_lossy+ret_img+to_webp/https://wisetoast.com/wp-content/uploads/2015/10/The-Persistence-of-Memory-salvador-deli-painting.jpg"))

# ╔═╡ cfcd39bc-007f-11eb-3306-c144b0d5ed6f
n_examples = min(200, size(img, 2))

# ╔═╡ ed827574-007f-11eb-184b-63c41a31a341
carved, marked_carved = shrink_n(img, n_examples);

# ╔═╡ d8124efa-007f-11eb-1918-81c1847d2753
@bind n Slider(1:length(carved))

# ╔═╡ 5ce8758c-0080-11eb-382e-0f25547c991f
hbox(img, marked_carved[n], sy=size(img))

# ╔═╡ Cell order:
# ╟─2eff7c8e-007f-11eb-337c-71e24bbe5b1d
# ╟─7bda3106-0080-11eb-3880-f911043ee036
# ╠═d8124efa-007f-11eb-1918-81c1847d2753
# ╟─5ce8758c-0080-11eb-382e-0f25547c991f
# ╟─4736c4a6-007f-11eb-1587-f773de9a54d1
# ╟─60a22fea-007f-11eb-0b85-61310e25b777
# ╟─643ec760-007f-11eb-02b6-879b064093a1
# ╟─9739002c-007f-11eb-34c6-cddee3fba7e0
# ╟─3badb08c-0080-11eb-094b-83d4ee955eb7
# ╟─9eb27932-007f-11eb-1eb8-a95c26681b65
# ╟─a6ea06ba-007f-11eb-3b29-895a03842e8d
# ╟─b0f4f2aa-007f-11eb-2732-9330dd587d4b
# ╟─b57ba6e0-007f-11eb-117f-cbf65a98ca47
# ╟─bdd0c206-007f-11eb-2030-cbb33969b289
# ╟─c667e778-007f-11eb-25ca-c55d58c6bab9
# ╟─c7396bc2-007f-11eb-13d6-61659dbccbdc
# ╟─cb0c5b4c-007f-11eb-2974-23387d6d1038
# ╟─0a370f74-0080-11eb-3fce-2f855eeb0058
# ╟─cfcd39bc-007f-11eb-3306-c144b0d5ed6f
# ╟─ed827574-007f-11eb-184b-63c41a31a341

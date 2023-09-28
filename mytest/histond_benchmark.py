import ROOT

ROOT.gRandom.SetSeed(123)
num_values = 10**5
num_bins = 10

df = ROOT.RDataFrame(num_values).Define("x", "gRandom->Uniform()").Define("y", "gRandom->Uniform()").Define("z", "gRandom->Uniform()")

# For 1D histograms
h1 = df.Histo1D(ROOT.RDF.TH1DModel("h1", "h1", num_bins, 0, 1), "x")
h1.GetValue()

# For 2D histograms
h2 = df.Histo2D(ROOT.RDF.TH2DModel("h2", "h2", num_bins, 0, 1, num_bins, 0, 1), "x", "y")
h2.GetValue()

# # For 3D histograms
h3 = df.Histo3D(ROOT.RDF.TH3DModel("h3", "h3", num_bins, 0, 1, num_bins, 0, 1, num_bins, 0, 1), "x", "y", "z")
h3.GetValue()

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "TCanvas.h"
#include "TH1D.h"
#include "TLatex.h"
#include "TLegend.h"
#include <Math/Vector4Dfwd.h>
#include <Math/GenVector/LorentzVector.h>
#include <Math/GenVector/PtEtaPhiM4D.h>
#include "TStyle.h"
#include <string>

#include "TStopwatch.h"

using namespace ROOT::VecOps;
using RNode = ROOT::RDF::RNode;
using cRVecF = const ROOT::RVecF &;

const auto z_mass = 91.2;

template<typename MyDF>
void print_info(MyDF df){
   auto entries = df.Count();
   std::cout << *entries << std::endl;

   if (0){
   auto colNames = df.GetColumnNames();
   // Print columns' names
   for (auto &&colName : colNames) std::cout << colName << " ";
   std::cout << "\n";
   }
}

void void_reco_zz_to_4l(cRVecF pt, cRVecF eta, cRVecF phi, cRVecF mass, const ROOT::RVecI & charge)
{
   RVec<RVec<size_t>> idx(2);
   idx[0].reserve(2); idx[1].reserve(2);

   if ((pt.size() <=1) | (eta.size() <=0) | (phi.size() <=0) | (mass.size() <=0) | (charge.size() <=0))
      return;

   // Find first lepton pair with invariant mass closest to Z mass
   auto idx_cmb = Combinations(pt, 2);
   //std::cout << pt.size() << " "  << eta.size() << " " << phi.size() <<  "\n";
   auto best_mass = -1;
   size_t best_i1 = 0; size_t best_i2 = 0;
   for (size_t i = 0; i < idx_cmb[0].size(); i++) {
      const auto i1 = idx_cmb[0][i];
      const auto i2 = idx_cmb[1][i];
      if (charge[i1] != charge[i2]) {
         ROOT::Math::PtEtaPhiMVector p1(pt[i1], eta[i1], phi[i1], mass[i1]);
         ROOT::Math::PtEtaPhiMVector p2(pt[i2], eta[i2], phi[i2], mass[i2]);
         const auto this_mass = (p1 + p2).M();
         if (std::abs(z_mass - this_mass) < std::abs(z_mass - best_mass)) {
            best_mass = this_mass;
            best_i1 = i1;
            best_i2 = i2;
         }
      }
   }
   idx[0].emplace_back(best_i1);
   idx[0].emplace_back(best_i2);

   // Reconstruct second Z from remaining lepton pair
   for (size_t i = 0; i < 4; i++) {
      if (i != best_i1 && i != best_i2) {
         idx[1].emplace_back(i);
      }
   }

   // Return indices of the pairs building two Z bosons
   return;
}

// Reconstruct two Z candidates from four leptons of the same kind
RVec<RVec<size_t>> reco_zz_to_4l(cRVecF pt, cRVecF eta, cRVecF phi, cRVecF mass, const ROOT::RVecI & charge)
{
   RVec<RVec<size_t>> idx(2);
   idx[0].reserve(2); idx[1].reserve(2);


   // Find first lepton pair with invariant mass closest to Z mass
   auto idx_cmb = Combinations(pt, 2);
   //std::cout << pt.size() << " "  << eta.size() << " " << phi.size() <<  "\n";
   auto best_mass = -1;
   size_t best_i1 = 0; size_t best_i2 = 0;
   for (size_t i = 0; i < idx_cmb[0].size(); i++) {
      const auto i1 = idx_cmb[0][i];
      const auto i2 = idx_cmb[1][i];
      if (charge[i1] != charge[i2]) {
         ROOT::Math::PtEtaPhiMVector p1(pt[i1], eta[i1], phi[i1], mass[i1]);
         ROOT::Math::PtEtaPhiMVector p2(pt[i2], eta[i2], phi[i2], mass[i2]);
         const auto this_mass = (p1 + p2).M();
         if (std::abs(z_mass - this_mass) < std::abs(z_mass - best_mass)) {
            best_mass = this_mass;
            best_i1 = i1;
            best_i2 = i2;
         }
      }
   }
   idx[0].emplace_back(best_i1);
   idx[0].emplace_back(best_i2);

   // Reconstruct second Z from remaining lepton pair
   for (size_t i = 0; i < 4; i++) {
      if (i != best_i1 && i != best_i2) {
         idx[1].emplace_back(i);
      }
   }

   // Return indices of the pairs building two Z bosons
   return idx;
}

// Reconstruct Higgs from four muons
RNode reco_higgs(RNode df)
{

   // Reconstruct Z systems
   auto df_z_idx =
      df.Define("Z_idx", reco_zz_to_4l, {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"});


   return df_z_idx;
}



void df103_NanoAODHiggsAnalysis(const bool run_fast = true)
{
   int N=3;
   // Enable multi-threading
   // ROOT::EnableImplicitMT();

   // In fast mode, take samples from */cms_opendata_2012_nanoaod_skimmed/*, which has
   // the preselections from the selection_* functions already applied.
   std::string path = "./rootfiles/";

   // Create dataframes for signal, background and data samples

   // Signal: Higgs -> 4 leptons
   std::vector<std::string> dataset1, dataset2, dataset3, dataset4, dataset5, dataset6;
   for (int i =0; i<N; i++){
      dataset1.push_back(path+"SMHiggsToZZTo4L.root");
      dataset2.push_back(path+"ZZTo4mu.root");
      dataset3.push_back(path+"ZZTo4e.root");
      dataset4.push_back(path+"ZZTo2e2mu.root");
      dataset5.push_back(path + "Run2012B_DoubleMuParked.root");
      dataset5.push_back(path + "Run2012C_DoubleMuParked.root");
      dataset6.push_back(path + "Run2012B_DoubleElectron.root");
      dataset6.push_back(path + "Run2012C_DoubleElectron.root");
   }
   // Background: ZZ -> 4 leptons
   // Note that additional background processes from the original paper with minor contribution were left out for this
   // tutorial.

    // {10,path + "SMHiggsToZZTo4L.root"};
   ROOT::RDataFrame df_sig_4l("Events", dataset1);
   ROOT::RDataFrame df_bkg_4mu("Events", dataset2);
   ROOT::RDataFrame df_bkg_4el("Events", dataset3);
   ROOT::RDataFrame df_bkg_2el2mu("Events", dataset4);
   // CMS data taken in 2012 (11.6 fb^-1 integrated luminosity)
   ROOT::RDataFrame df_data_doublemu(
      "Events", dataset5);
   ROOT::RDataFrame df_data_doubleel(
      "Events", dataset6);

   print_info(df_sig_4l);
   print_info(df_bkg_4mu);
   print_info(df_bkg_4el);
   print_info(df_bkg_2el2mu);
   print_info(df_data_doublemu);
   print_info(df_data_doublemu);

   auto df_sig_4mu_reco = reco_higgs(df_sig_4l);

    
   TStopwatch timer;
   timer.Start();

   std::vector<std::string> columns = {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"};

   df_sig_4l.Foreach(void_reco_zz_to_4l, columns);
   
   std::cout << "Foreach time " << timer.RealTime() << std::endl;

   timer.Stop();


/*
    auto df_h_sig_4mu = df_sig_4mu_reco
         .Histo1D({"MyHisto", "", 100, 70, 180}, "Muon_phi");
    auto df_sig_4el_reco = reco_higgs(df_sig_4l);
    auto df_h_bkg_4mu = df_sig_4el_reco
         .Histo1D({"MyHisto", "", 100, 70, 180}, "Muon_phi");
    auto df_bkg_4mu_reco = reco_higgs_to_4mu(df_bkg_4mu);
    auto df_bkg_4el_reco = reco_higgs_to_4el(df_bkg_4el);
    
    auto df_data_4mu_reco = reco_higgs_to_4mu(df_data_doublemu);
    auto df_data_4el_reco = reco_higgs_to_4el(df_data_doubleel);
    
    auto df_sig_2el2mu_reco = reco_higgs_to_2el2mu(df_sig_4l);
    auto df_bkg_2el2mu_reco = reco_higgs_to_2el2mu(df_bkg_2el2mu);
    auto df_data_2el2mu_reco = reco_higgs_to_2el2mu(df_data_doublemu);
   auto i = ROOT::RDF::RunGraphs({df_h_sig_4mu, df_h_bkg_4mu});
   
   std::cout << i << "\n";
  
 */


    //ROOT::RDF::SaveGraph(df_h_sig_4mu, "graph.txt");
    /* ({df_h_sig_4mu, df_h_bkg_4mu, df_h_data_4mu,
                         df_h_sig_4el, df_h_bkg_4el, df_h_data_4el,
                         df_h_sig_2el2mu, df_h_bkg_2el2mu, df_h_data_2el2mu});
 */}

int main()
{
   df103_NanoAODHiggsAnalysis(/*fast=*/true);
}

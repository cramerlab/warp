using M.Controls.Sociology.Dialogs;
using MahApps.Metro.Controls.Dialogs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Warp.Sociology;

namespace M.Controls.Sociology
{
    /// <summary>
    /// Interaction logic for SpeciesView.xaml
    /// </summary>
    public partial class SpeciesView : UserControl
    {
        public Species Species
        {
            get { return (Species)GetValue(SpeciesProperty); }
            set { SetValue(SpeciesProperty, value); }
        }
        public static readonly DependencyProperty SpeciesProperty = DependencyProperty.Register("Species", typeof(Species), typeof(SpeciesView), new PropertyMetadata(null, (sender, args) => ((SpeciesView)sender).DataContext = args.NewValue));
        

        public SpeciesView()
        {
            InitializeComponent();
        }

        private void ButtonParticles_Click(object sender, RoutedEventArgs e)
        {
            MenuParticles.IsOpen = true;
        }

        private async void AddParticles_Click(object sender, RoutedEventArgs e)
        {
            MainWindow Window = (MainWindow)Application.Current.MainWindow;
            Species S = Species;

            CustomDialog NewDialog = new CustomDialog();
            NewDialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogSpeciesParticleSets NewDialogContent = new DialogSpeciesParticleSets(Window.ActivePopulation, S);
            NewDialogContent.Close += async () => await Window.HideMetroDialogAsync(NewDialog);

            NewDialogContent.Add += async () =>
            {
                await Window.HideMetroDialogAsync(NewDialog);

                var NewSpeciesProgress = await Window.ShowProgressAsync("Please wait while particle statistics are updated...",
                                                                        "");
                NewSpeciesProgress.SetIndeterminate();
                
                await Task.Run(() =>
                {
                    S.AddParticles(NewDialogContent.ParticlesFinal);

                    S.CalculateParticleStats();

                    S.Commit();
                    S.Save();
                });

                await NewSpeciesProgress.CloseAsync();
            };

            NewDialog.Content = NewDialogContent;
            await Window.ShowMetroDialogAsync(NewDialog);
        }

        private async void ExportSubtomo_Click(object sender, RoutedEventArgs e)
        {
            MainWindow Window = (MainWindow)Application.Current.MainWindow;
            Species S = Species;

            CustomDialog NewDialog = new CustomDialog();
            NewDialog.HorizontalContentAlignment = HorizontalAlignment.Center;

            DialogTomoParticleExport NewDialogContent = new DialogTomoParticleExport(Window.ActivePopulation, S);
            NewDialogContent.Close += async () => await Window.HideMetroDialogAsync(NewDialog);
            
            NewDialog.Content = NewDialogContent;
            await Window.ShowMetroDialogAsync(NewDialog);
        }
    }
}
